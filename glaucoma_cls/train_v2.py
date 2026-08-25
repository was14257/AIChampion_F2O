"""v2 classifier: concept 9개(SEG6+OCT_CONCEPTS3, GAMMA 172장 학습) 대신
concept 11개(SEG6+RNFL5, GRAPE 244장 실측 학습)를 쓰고, 데이터도
GAMMA_train+REFUGE+GRAPE 전체를 하나의 pool로 묶어 glaucoma 비율을 유지한
stratified 5-fold CV로 학습/평가한다 (2026-08-09, 사용자 요청).

기존 train.py(GAMMA_train만 val, leakage 방지 원칙)와 달리 val도 전체 pool
에서 stratified로 뽑는다 - GRAPE를 훈련에만 넣었을 때 specificity가
0.11~0.50까지 급락하던 문제가, concept을 11개로 늘리고 평가를 pool 전체
stratified로 바꾸자 spec 0.85~0.90으로 회복됨을 확인(temp/grape_posweight_sweep.py,
temp/grape_cls_with_concept.py 대비)."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, roc_curve

from config import CFG
from glaucoma_cls.data import loaders, class_weights, build_pool_v2, load_concept_table_v2, attach_concepts
from glaucoma_cls.model import GlaucomaNet

OUT_DIR = CFG.paths.output_root / "glaucoma_cls"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 10
BATCH_SIZE = 64
LR_ENC, LR_HEAD = 1e-5, 1e-3
UNFREEZE_LAST_N = 8
CONCEPT_PROJ_DIM = 32
POS_WEIGHT_SCALE = 0.6
N_FOLDS = 5
SEED = 42


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device, return_probs=False):
    model.eval()
    probs, labels = [], []
    for x, c, y in loader:
        x = x.to(device); c = c.to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x, c))
        probs.append(p.float().cpu().numpy())
        labels.append(np.asarray(y))
    probs = np.concatenate(probs); labels = np.concatenate(labels)
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    pred = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    macro_f1 = f1_score(labels, pred, average="macro", zero_division=0)
    if len(set(labels)) > 1:
        fpr, tpr, thr = roc_curve(labels, probs)
        thr_opt = float(thr[np.argmax(tpr - fpr)])
    else:
        thr_opt = 0.5
    pred_opt = (probs >= thr_opt).astype(int)
    macro_f1_opt = f1_score(labels, pred_opt, average="macro", zero_division=0)
    out = {"auc": auc, "acc": accuracy_score(labels, pred), "macro_f1": macro_f1,
           "macro_f1_opt": macro_f1_opt, "thr_opt": thr_opt,
           "sens": tp / max(1, tp + fn), "spec": tn / max(1, tn + fp)}
    if return_probs:
        out["probs"] = probs
        out["labels"] = labels
    return out


def run_fold(fold_idx, train_df, val_df, concept_table, n_concepts, device):
    set_seed(SEED + fold_idx)
    train_df, val_df = attach_concepts(train_df, val_df, concept_table, n_concepts)
    model = GlaucomaNet(freeze_encoder=True, unfreeze_last_n=UNFREEZE_LAST_N,
                        n_concepts=n_concepts, concept_proj_dim=CONCEPT_PROJ_DIM).to(device)
    tr, va = loaders(train_df, val_df, batch_size=BATCH_SIZE, use_concepts=True)
    pos_w = torch.tensor([class_weights(train_df) * POS_WEIGHT_SCALE], device=device)
    opt = torch.optim.AdamW(model.param_groups(LR_ENC, LR_HEAD), weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_auc, best_macro_f1 = -1, -1
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, c, y in tr:
            x = x.to(device); c = c.to(device); y = y.float().to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model(x, c), y, pos_weight=pos_w)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()
        m = evaluate(model, va, device)
        best_auc = max(best_auc, m["auc"])
        best_macro_f1 = max(best_macro_f1, m["macro_f1_opt"])
    final_m = evaluate(model, va, device, return_probs=True)
    return {"best_auc": best_auc, "final_auc": final_m["auc"],
            "best_macro_f1": best_macro_f1, "final_macro_f1": final_m["macro_f1"],
            "final_macro_f1_opt": final_m["macro_f1_opt"], "thr_opt": final_m["thr_opt"],
            "final_sens": final_m["sens"], "final_spec": final_m["spec"],
            "n_train": len(train_df), "n_val": len(val_df),
            "oof_probs": final_m["probs"], "oof_labels": final_m["labels"],
            "oof_paths": val_df["path"].tolist()}, model


def main():
    pool = build_pool_v2(seed=SEED)
    print(f"pool total={len(pool)}  by dataset={pool['dataset'].value_counts().to_dict()}  "
          f"glaucoma_ratio={pool['label'].mean():.3f}")

    concept_table, n_concepts = load_concept_table_v2()
    print(f"n_concepts={n_concepts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    results = []
    best_model_state = None
    best_auc_overall = -1
    all_oof_probs, all_oof_labels, all_oof_paths = [], [], []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(pool, pool["label"])):
        t0 = time.time()
        train_df = pool.iloc[tr_idx].reset_index(drop=True)
        val_df = pool.iloc[va_idx].reset_index(drop=True)
        r, model = run_fold(fold_idx, train_df, val_df, concept_table, n_concepts, device)
        r.update({"fold_idx": fold_idx, "elapsed": round(time.time() - t0, 1)})
        print(f"[fold={fold_idx}] best_auc={r['best_auc']:.3f} final_auc={r['final_auc']:.3f} "
              f"macro_f1_opt={r['final_macro_f1_opt']:.3f} sens={r['final_sens']:.2f} "
              f"spec={r['final_spec']:.2f} n_train={r['n_train']} n_val={r['n_val']} ({r['elapsed']}s)")
        all_oof_probs.append(r.pop("oof_probs"))
        all_oof_labels.append(r.pop("oof_labels"))
        all_oof_paths.extend(r.pop("oof_paths"))
        results.append(r)
        if r["best_auc"] > best_auc_overall:
            best_auc_overall = r["best_auc"]
            best_model_state = model.state_dict()

    aucs = [r["best_auc"] for r in results]
    f1s = [r["best_macro_f1"] for r in results]
    specs = [r["final_spec"] for r in results]
    print(f"\n=== GAMMA+REFUGE+GRAPE pool stratified 5-fold (11 concepts) ===")
    print(f"auc={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
          f"macro_f1_opt={np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
          f"spec={np.mean(specs):.4f}±{np.std(specs):.4f}")
    print(f"folds_auc={aucs}")
    print(f"folds_spec={specs}")

    out_json = OUT_DIR / "cv_results_v2_11concept_pool.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_json}")

    oof_probs = np.concatenate(all_oof_probs)
    oof_labels = np.concatenate(all_oof_labels)
    oof_path = OUT_DIR / "oof_v2_11concept.npz"
    np.savez(oof_path, probs=oof_probs, labels=oof_labels, paths=np.array(all_oof_paths))
    print(f"Saved OOF: {oof_path}  (n={len(oof_probs)}, auc={roc_auc_score(oof_labels, oof_probs):.4f})")

    ckpt_path = OUT_DIR / "glaucoma_v2_11concept_bestfold.pth"
    torch.save({"model": best_model_state, "n_concepts": n_concepts,
                "concept_proj_dim": CONCEPT_PROJ_DIM, "unfreeze_last_n": UNFREEZE_LAST_N,
                "concepts": list(np.load(CFG.paths.oct_features / "cbm_concepts_v2.npz",
                                         allow_pickle=True)["concepts"])},
               ckpt_path)
    print(f"Saved best-fold model: {ckpt_path} (best_auc={best_auc_overall:.4f})")


if __name__ == "__main__":
    main()

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, f1_score
from torch.utils.data import DataLoader

from glaucoma_cls.data import (build_frames, loaders, class_weights, FundusDS,
                               load_concept_table, attach_concepts,
                               attach_concepts_external)
from glaucoma_cls.model import GlaucomaNet
from local_config import (CLS_EPOCHS, CLS_BATCH_SIZE, CLS_LR_ENC, CLS_LR_HEAD,
                          CLS_FREEZE, CLS_UNFREEZE_LAST_N, CLS_ADD_GAMMA_TRAIN,
                          CLS_PREDICT_ONLY, CLS_USE_CONCEPTS, CLS_POS_WEIGHT_SCALE,
                          CLS_CONCEPT_PROJ_DIM)

_GAMMA_TEST_CONCEPTS_NPZ = Path(
    "C:/Users/hogri/OneDrive/Desktop/AIGS자율공모/Code/outputs/oct_features/cbm_concepts_gamma_test.npz")


def load_gamma_test_concepts():
    """GAMMA test's (labels hidden) 100 concepts -> dict keyed by case_id.
    Stored separately from cbm_concepts.npz (train/val) - see extract_concepts_gamma_test.py."""
    d = np.load(_GAMMA_TEST_CONCEPTS_NPZ, allow_pickle=True)
    return dict(zip(d["case_ids"].tolist(), d["X"].astype("float32")))


def attach_gamma_test_concepts(test_df, n_concepts, mean, std):
    """Attaches concepts to GAMMA test using the train fold's mean/std (no labels, so a separate npz)."""
    table = load_gamma_test_concepts()
    return attach_concepts_external(test_df, table, n_concepts, mean, std,
                                    key_col="case_id")

OUT = Path("C:/Users/hogri/OneDrive/Desktop/AIGS자율공모/Code/outputs")
OUT = OUT if OUT.parent.exists() else Path("/home/tta/outputs")
OUT = OUT / "glaucoma_cls"
CKPT = OUT / "best.pth"


def set_seed(s=42):
    """Seeds all RNGs (python/numpy/torch) for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device, use_concepts=False):
    """Runs the model over a loader and returns AUC/acc/macro-F1/sens/spec (at threshold 0.5) plus raw probs."""
    model.eval()
    probs, labels = [], []
    for batch in loader:
        if use_concepts:
            x, c, y = batch
            x = x.to(device); c = c.to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                p = torch.sigmoid(model(x, c))
        else:
            x, y = batch
            x = x.to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                p = torch.sigmoid(model(x))
        probs.append(p.float().cpu().numpy())
        labels.append(np.asarray(y))
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    pred = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    macro_f1 = f1_score(labels, pred, average="macro", zero_division=0)
    return {"auc": auc, "acc": acc, "macro_f1": macro_f1, "sens": sens, "spec": spec}, probs


def main():
    """Trains (unless CLS_PREDICT_ONLY) GlaucomaNet, picks the best checkpoint by
    (AUC, macro_f1), calibrates a threshold on GAMMA val, and writes GAMMA test pseudo-labels."""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    ext, val, test = build_frames(use_datasets=("REFUGE",))
    print(f"Train {len(ext)} / Val (GAMMA train) {len(val)} / Target (GAMMA test) {len(test)}")

    if CLS_ADD_GAMMA_TRAIN:
        ext = pd.concat([ext, val[["path", "label", "dataset"]]], ignore_index=True)
        print(f"  GAMMA train included -> train {len(ext)}")

    n_concepts = 0
    concept_mean = concept_std = None
    if CLS_USE_CONCEPTS:
        concept_table, n_concepts = load_concept_table()
        ext, val, concept_mean, concept_std = attach_concepts(
            ext, val, concept_table, n_concepts, return_stats=True)
        print(f"  {n_concepts} concepts combined (CDR etc., confirmed via 5-fold CV on 2026-07-24)")

    model = GlaucomaNet(freeze_encoder=CLS_FREEZE, unfreeze_last_n=CLS_UNFREEZE_LAST_N,
                        n_concepts=n_concepts, concept_proj_dim=CLS_CONCEPT_PROJ_DIM).to(device)
    ngpu = torch.cuda.device_count()
    if ngpu > 1 and not CLS_PREDICT_ONLY:
        model = torch.nn.DataParallel(model)
    _m = model.module if isinstance(model, torch.nn.DataParallel) else model

    if not CLS_PREDICT_ONLY:
        tr, va = loaders(ext, val, batch_size=CLS_BATCH_SIZE, use_concepts=CLS_USE_CONCEPTS)
        pos_w = torch.tensor([class_weights(ext) * CLS_POS_WEIGHT_SCALE], device=device)
        opt = torch.optim.AdamW(_m.param_groups(CLS_LR_ENC, CLS_LR_HEAD), weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CLS_EPOCHS)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

        best = (-1, -1)  # (auc, macro_f1) - when auc ties (e.g. multiple
                         # epochs hit AUC=1.0 on the n=20 val), pick the
                         # better epoch by macro_f1. Previously only auc was
                         # checked, so it stopped at the first epoch to hit
                         # AUC=1.0 and saved a checkpoint with a lower macro_f1.
        for ep in range(1, CLS_EPOCHS + 1):
            model.train()
            t0 = time.time()
            tot = 0
            for batch in tr:
                if CLS_USE_CONCEPTS:
                    x, c, y = batch
                    x = x.to(device); c = c.to(device); y = y.float().to(device)
                    opt.zero_grad()
                    with torch.autocast("cuda", enabled=device.type == "cuda"):
                        loss = F.binary_cross_entropy_with_logits(model(x, c), y, pos_weight=pos_w)
                else:
                    x, y = batch
                    x = x.to(device); y = y.float().to(device)
                    opt.zero_grad()
                    with torch.autocast("cuda", enabled=device.type == "cuda"):
                        loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_w)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item()
            sched.step()
            m, _ = evaluate(model, va, device, use_concepts=CLS_USE_CONCEPTS)
            print(f"[{ep:2d}/{CLS_EPOCHS}] loss={tot/len(tr):.4f}  "
                  f"GAMMA-val AUC={m['auc']:.3f} macro_f1={m['macro_f1']:.3f} acc={m['acc']:.3f} "
                  f"sens={m['sens']:.2f} spec={m['spec']:.2f} ({time.time()-t0:.0f}s)")
            score = (m["auc"], m["macro_f1"])
            if score > best:
                best = score
                torch.save({"model": _m.state_dict(), "val": m, "epoch": ep}, CKPT)
                print(f"   -> saved best (AUC={best[0]:.3f} macro_f1={best[1]:.3f})")
        print(f"\nTraining done. best GAMMA-val AUC={best[0]:.3f} macro_f1={best[1]:.3f}")

    ck = torch.load(CKPT, map_location=device, weights_only=False)
    _m.load_state_dict(ck["model"])
    _m.to(device)
    print(f"Loaded best checkpoint (val AUC={ck['val']['auc']:.3f})")

    val_loader = DataLoader(FundusDS(val, train=False, use_concepts=CLS_USE_CONCEPTS),
                            batch_size=CLS_BATCH_SIZE, num_workers=0)
    vm, vprobs = evaluate(_m, val_loader, device, use_concepts=CLS_USE_CONCEPTS)
    vlabels = val["label"].values
    fpr, tpr, thr = roc_curve(vlabels, vprobs)
    thr_opt = float(thr[np.argmax(tpr - fpr)])
    vpred = (vprobs >= thr_opt).astype(int)
    tn, fp, fn, tp = confusion_matrix(vlabels, vpred, labels=[0, 1]).ravel()
    print(f"Calibrated threshold={thr_opt:.3f} | GAMMA-val (post-calibration) "
          f"acc={(tp+tn)/len(vlabels):.3f} sens={tp/max(1,tp+fn):.2f} spec={tn/max(1,tn+fp):.2f}")

    if CLS_USE_CONCEPTS:
        test = attach_gamma_test_concepts(test, n_concepts, concept_mean, concept_std)
    test_loader = DataLoader(FundusDS(test, train=False, with_label=False, use_concepts=CLS_USE_CONCEPTS),
                             batch_size=CLS_BATCH_SIZE, num_workers=0)
    _m.eval()
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            if CLS_USE_CONCEPTS:
                x, c, cid = batch
                x = x.to(device); c = c.to(device)
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    p = torch.sigmoid(_m(x, c)).float().cpu().numpy()
            else:
                x, cid = batch
                x = x.to(device)
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    p = torch.sigmoid(_m(x)).float().cpu().numpy()
            for c_, pr in zip(cid, p):
                rows.append({"case_id": c_, "glaucoma_prob": round(float(pr), 4),
                             "pseudo_label": int(pr >= thr_opt)})
    df = pd.DataFrame(rows)
    out_csv = OUT / "gamma_test_pseudolabels.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved pseudo-labels: {out_csv}")
    print("Distribution:", df["pseudo_label"].value_counts().to_dict(),
          "| ambiguous (0.4~0.6):", int(df["glaucoma_prob"].between(0.4, 0.6).sum()))


if __name__ == "__main__":
    main()

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                             f1_score, roc_curve)

from glaucoma_cls.data import (build_frames, loaders, class_weights,
                               load_concept_table, attach_concepts)
from glaucoma_cls.model import GlaucomaNet

OUT = Path("C:/Users/hogri/OneDrive/Desktop/AIGS자율공모/Code/outputs/glaucoma_cls")
OUT.mkdir(parents=True, exist_ok=True)

EPOCHS = 10
BATCH_SIZE = 64
LR_ENC, LR_HEAD = 1e-5, 1e-3
UNFREEZE_OPTIONS = [0, 4]
SEEDS = [0, 1, 2, 3, 4]

_BASE_CONDITIONS = [
    {"name": "all_with_gamma",    "use_datasets": ("REFUGE", "ORIGA", "G1020"), "include_gamma_train": True},
    {"name": "all_no_gamma",      "use_datasets": ("REFUGE", "ORIGA", "G1020"), "include_gamma_train": False},
    {"name": "asian_with_gamma",  "use_datasets": ("REFUGE", "ORIGA"),          "include_gamma_train": True},
    {"name": "asian_no_gamma",    "use_datasets": ("REFUGE", "ORIGA"),          "include_gamma_train": False},
]
CONDITIONS = [
    {**c, "name": f"{c['name']}_unf{n}", "unfreeze_last_n": n}
    for c in _BASE_CONDITIONS for n in UNFREEZE_OPTIONS
]

# Compares only with/without combining concepts (CDR/ilm_rough etc., 9 total)
# under the best condition (all_with_gamma, unf4).
# 1st experiment (2026-07-23): no_concept 0.747+-0.079 -> with_concept 0.963+-0.015, a big improvement.
# Separately confirmed concept (especially CDR) alone is a strong signal on GAMMA too (0.947 with plain logistic regression).
CONCEPT_CONDITIONS = [
    {"name": "best_no_concept", "use_datasets": ("REFUGE", "ORIGA", "G1020"),
     "include_gamma_train": True, "unfreeze_last_n": 4, "use_concepts": False},
    {"name": "best_with_concept", "use_datasets": ("REFUGE", "ORIGA", "G1020"),
     "include_gamma_train": True, "unfreeze_last_n": 4, "use_concepts": True},
]

# Fixes concept combination and sweeps only the unfreeze width (checking room
# for embedding-side improvement).
# 2nd experiment (2026-07-23, after fixing the LayerNorm bug): unf0~8 all get
# auc 0.96~0.97, macro_f1(opt_thr) improved greatly to 0.91~0.94. But
# macro_f1(thr=0.5) is still 0.429(sens=1.0/spec=0.09) - discrimination
# improved, but pos_weight still skews probabilities toward positive, so it's
# still bad at the fixed deployment threshold=0.5.
CONCEPT_UNFREEZE_CONDITIONS = [
    {"name": f"concept_unf{n}", "use_datasets": ("REFUGE", "ORIGA", "G1020"),
     "include_gamma_train": True, "unfreeze_last_n": n, "use_concepts": True}
    for n in [0, 2, 4, 8]
]

# Fixes the best architecture (unf8+concept), sweeps only pos_weight scale -
# an experiment to fix probability calibration itself so classification is
# normal even at threshold=0.5.
# 1st run (2026-07-23): scale 1.0->0.429, 0.5->0.671, 0.33->0.811, 0.2->0.849 (macro_f1, thr=.5)
# AUC doesn't drop even down to 0.2 (0.969) - search for the optimum by lowering scale further.
POS_WEIGHT_CONDITIONS = [
    {"name": f"posw_scale{s}", "use_datasets": ("REFUGE", "ORIGA", "G1020"),
     "include_gamma_train": True, "unfreeze_last_n": 8, "use_concepts": True,
     "pos_weight_scale": s}
    for s in [1.0, 0.5, 0.33, 0.2]
]

POS_WEIGHT_CONDITIONS_V2 = [
    {"name": f"posw_scale{s}", "use_datasets": ("REFUGE", "ORIGA", "G1020"),
     "include_gamma_train": True, "unfreeze_last_n": 8, "use_concepts": True,
     "pos_weight_scale": s}
    for s in [0.2, 0.15, 0.1, 0.05]
]

# Re-validates the final best configuration (unf8+concept+pos_weight_scale=0.2).
# After fixing the minor leak where attach_concepts() normalized using
# statistics that mixed in the fold's val set too (now computes mean/std
# from fold train only), checks whether the same configuration is still good.
BEST_CONFIG_RECHECK = [
    {"name": "best_recheck", "use_datasets": ("REFUGE", "ORIGA", "G1020"),
     "include_gamma_train": True, "unfreeze_last_n": 8, "use_concepts": True,
     "pos_weight_scale": 0.2},
]

# Validates the concern (raised by the user, 2026-07-24) that concatenating
# concept (9-dim) raw is too small relative to the embedding (1024-dim),
# potentially causing asymmetric dropout/weight_decay. Expands concept via a
# small dedicated MLP up to proj_dim before concatenation (model.py's
# concept_proj_dim). proj_dim=0 keeps the existing behavior (raw concat, the
# best so far).
CONCEPT_PROJ_CONDITIONS = [
    {"name": f"concept_proj{d}", "use_datasets": ("REFUGE", "ORIGA", "G1020"),
     "include_gamma_train": True, "unfreeze_last_n": 8, "use_concepts": True,
     "pos_weight_scale": 0.2, "concept_proj_dim": d}
    for d in [0, 16, 32, 64]
]

# Increases n by mixing in stratified samples from REFUGE/ORIGA/G1020 into
# val, not just GAMMA (n=20) (per user request, 2026-07-24) - GAMMA alone
# makes per-fold AUC swing a lot (0.875~1.0), so this aims to reduce
# validation variance. Splits off ext_val_frac from each external dataset
# and merges into val.
EXT_VAL_CONDITIONS = [
    {"name": f"extval{int(f*100)}pct", "use_datasets": ("REFUGE", "ORIGA", "G1020"),
     "include_gamma_train": True, "unfreeze_last_n": 8, "use_concepts": True,
     "pos_weight_scale": 0.2, "ext_val_frac": f}
    for f in [0.0, 0.05, 0.1, 0.2]
]


def set_seed(s):
    """Seeds all RNGs (python/numpy/torch) for reproducibility."""
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device, use_concepts=False):
    """Runs the model over a loader and returns AUC/acc/macro-F1 (at 0.5 and
    at the Youden's J optimal threshold) plus sensitivity/specificity."""
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
    probs = np.concatenate(probs); labels = np.concatenate(labels)
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    pred = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    macro_f1 = f1_score(labels, pred, average="macro", zero_division=0)
    # threshold=0.5 distorts F1 if pos_weight-corrected training skews
    # probabilities to one side (collapsing to sens=1.0, spec=0). Also
    # computes macro_f1 at a Youden's J (max tpr-fpr) corrected threshold -
    # to distinguish cases where AUC (ranking) is fine but F1 (absolute
    # probability) alone is low.
    if len(set(labels)) > 1:
        fpr, tpr, thr = roc_curve(labels, probs)
        thr_opt = float(thr[np.argmax(tpr - fpr)])
    else:
        thr_opt = 0.5
    pred_opt = (probs >= thr_opt).astype(int)
    macro_f1_opt = f1_score(labels, pred_opt, average="macro", zero_division=0)
    return {"auc": auc, "acc": accuracy_score(labels, pred), "macro_f1": macro_f1,
            "macro_f1_opt": macro_f1_opt, "thr_opt": thr_opt,
            "sens": tp / max(1, tp + fn), "spec": tn / max(1, tn + fp)}


def run_one(seed, use_datasets, include_gamma_train, unfreeze_last_n, device,
            use_concepts=False, pos_weight_scale=1.0, fold_idx=None, n_folds=5,
            concept_proj_dim=0, ext_val_frac=0.0):
    """Trains one GlaucomaNet run for a given condition/seed/fold and returns
    best/final metrics on the val split."""
    set_seed(seed)
    ext, val, _ = build_frames(gamma_val_frac=0.2, seed=seed,
                               include_gamma_train=include_gamma_train,
                               use_datasets=use_datasets,
                               fold_idx=fold_idx, n_folds=n_folds,
                               ext_val_frac=ext_val_frac)
    n_concepts = 0
    if use_concepts:
        concept_table, n_concepts = load_concept_table()
        ext, val = attach_concepts(ext, val, concept_table, n_concepts)
    model = GlaucomaNet(freeze_encoder=True, unfreeze_last_n=unfreeze_last_n,
                        n_concepts=n_concepts, concept_proj_dim=concept_proj_dim).to(device)
    tr, va = loaders(ext, val, batch_size=BATCH_SIZE, use_concepts=use_concepts)
    pos_w = torch.tensor([class_weights(ext) * pos_weight_scale], device=device)
    opt = torch.optim.AdamW(model.param_groups(LR_ENC, LR_HEAD), weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_auc = -1
    best_macro_f1 = -1
    for ep in range(1, EPOCHS + 1):
        model.train()
        for batch in tr:
            if use_concepts:
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
            scaler.step(opt); scaler.update()
        sched.step()
        m = evaluate(model, va, device, use_concepts=use_concepts)
        best_auc = max(best_auc, m["auc"])
        best_macro_f1 = max(best_macro_f1, m["macro_f1_opt"])
    final_m = evaluate(model, va, device, use_concepts=use_concepts)
    return {"best_auc": best_auc, "final_auc": final_m["auc"],
            "best_macro_f1": best_macro_f1, "final_macro_f1": final_m["macro_f1"],
            "final_macro_f1_opt": final_m["macro_f1_opt"], "thr_opt": final_m["thr_opt"],
            "final_sens": final_m["sens"], "final_spec": final_m["spec"],
            "n_train": len(ext), "n_val": len(val)}


def _run_sweep(conditions, out_name, use_concepts_key=None):
    """Runs run_one() over all (condition, seed) combos, prints per-run and
    per-condition summaries, and saves raw results as JSON to OUT/out_name."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for cond in conditions:
        for seed in SEEDS:
            t0 = time.time()
            use_concepts = cond.get("use_concepts", False)
            pos_weight_scale = cond.get("pos_weight_scale", 1.0)
            concept_proj_dim = cond.get("concept_proj_dim", 0)
            r = run_one(seed, cond["use_datasets"], cond["include_gamma_train"],
                        cond["unfreeze_last_n"], device, use_concepts=use_concepts,
                        pos_weight_scale=pos_weight_scale, concept_proj_dim=concept_proj_dim)
            r.update({"condition": cond["name"], "seed": seed,
                      "elapsed": round(time.time() - t0, 1)})
            print(f"[{cond['name']:<18} seed={seed}] "
                  f"best_auc={r['best_auc']:.3f} final_auc={r['final_auc']:.3f} "
                  f"best_macro_f1(opt_thr)={r['best_macro_f1']:.3f} "
                  f"final_macro_f1(thr=0.5)={r['final_macro_f1']:.3f} "
                  f"final_macro_f1_opt={r['final_macro_f1_opt']:.3f} thr_opt={r['thr_opt']:.3f} "
                  f"sens={r['final_sens']:.2f} spec={r['final_spec']:.2f} "
                  f"n_train={r['n_train']} ({r['elapsed']}s)")
            results.append(r)

    out_json = OUT / out_name
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Per-condition summary (mean ± std) ===")
    for cond in conditions:
        aucs = [r["best_auc"] for r in results if r["condition"] == cond["name"]]
        f1s = [r["best_macro_f1"] for r in results if r["condition"] == cond["name"]]
        f1s_thr05 = [r["final_macro_f1"] for r in results if r["condition"] == cond["name"]]
        print(f"{cond['name']:<18} auc={np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
              f"macro_f1(opt_thr)={np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
              f"macro_f1(thr=0.5)={np.mean(f1s_thr05):.3f}±{np.std(f1s_thr05):.3f}  "
              f"folds_auc={aucs}")

    print(f"\nSaved: {out_json}")


def _run_true_kfold(conditions, out_name, n_folds=5, seed=0):
    """seed (2026-07-24): until now, the 5 repeats each independently drew a
    random 80/20 (build_frames's fold_idx=None path), so the same 20 could
    be drawn multiple times or never at all - re-validates with a true
    non-overlapping k-fold (each of GAMMA's 100 images in val exactly once)
    to reduce the n=20 val overfitting/chance-split problem."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for cond in conditions:
        for fold_idx in range(n_folds):
            t0 = time.time()
            use_concepts = cond.get("use_concepts", False)
            pos_weight_scale = cond.get("pos_weight_scale", 1.0)
            concept_proj_dim = cond.get("concept_proj_dim", 0)
            ext_val_frac = cond.get("ext_val_frac", 0.0)
            r = run_one(seed, cond["use_datasets"], cond["include_gamma_train"],
                        cond["unfreeze_last_n"], device, use_concepts=use_concepts,
                        pos_weight_scale=pos_weight_scale, fold_idx=fold_idx, n_folds=n_folds,
                        concept_proj_dim=concept_proj_dim, ext_val_frac=ext_val_frac)
            r.update({"condition": cond["name"], "fold_idx": fold_idx,
                      "elapsed": round(time.time() - t0, 1)})
            print(f"[{cond['name']:<18} fold={fold_idx}] "
                  f"best_auc={r['best_auc']:.3f} final_auc={r['final_auc']:.3f} "
                  f"best_macro_f1(opt_thr)={r['best_macro_f1']:.3f} "
                  f"final_macro_f1(thr=0.5)={r['final_macro_f1']:.3f} "
                  f"sens={r['final_sens']:.2f} spec={r['final_spec']:.2f} "
                  f"n_train={r['n_train']} ({r['elapsed']}s)")
            results.append(r)

    out_json = OUT / out_name
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Per-condition summary (true k-fold, mean ± std) ===")
    for cond in conditions:
        aucs = [r["best_auc"] for r in results if r["condition"] == cond["name"]]
        f1s = [r["best_macro_f1"] for r in results if r["condition"] == cond["name"]]
        f1s_thr05 = [r["final_macro_f1"] for r in results if r["condition"] == cond["name"]]
        print(f"{cond['name']:<18} auc={np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
              f"macro_f1(opt_thr)={np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
              f"macro_f1(thr=0.5)={np.mean(f1s_thr05):.3f}±{np.std(f1s_thr05):.3f}  "
              f"folds_auc={aucs}")

    print(f"\nSaved: {out_json}")


def main_true_kfold():
    """Entry point: re-validate the best config with true non-overlapping k-fold."""
    _run_true_kfold(BEST_CONFIG_RECHECK, "cv_results_true_kfold.json")


def main_concept_proj():
    """Entry point: sweep concept_proj_dim under true k-fold."""
    _run_true_kfold(CONCEPT_PROJ_CONDITIONS, "cv_results_concept_proj.json")


def main_ext_val():
    """Entry point: sweep ext_val_frac under true k-fold."""
    _run_true_kfold(EXT_VAL_CONDITIONS, "cv_results_ext_val.json")


def main():
    """Entry point: the base ethnicity/domain ablation sweep."""
    _run_sweep(CONDITIONS, "cv_results.json")


def main_concept():
    """Entry point: with vs. without concept features."""
    _run_sweep(CONCEPT_CONDITIONS, "cv_results_concept.json")


def main_concept_unfreeze():
    """Entry point: sweep unfreeze width with concepts fixed on."""
    _run_sweep(CONCEPT_UNFREEZE_CONDITIONS, "cv_results_concept_unfreeze.json")


def main_pos_weight():
    """Entry point: sweep pos_weight_scale."""
    _run_sweep(POS_WEIGHT_CONDITIONS, "cv_results_pos_weight.json")


def main_pos_weight_v2():
    """Entry point: finer pos_weight_scale sweep."""
    _run_sweep(POS_WEIGHT_CONDITIONS_V2, "cv_results_pos_weight_v2.json")


def main_best_recheck():
    """Entry point: re-validate the best config (non-true-kfold sweep path)."""
    _run_sweep(BEST_CONFIG_RECHECK, "cv_results_best_recheck.json")


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "concept":
        main_concept()
    elif arg == "concept_unfreeze":
        main_concept_unfreeze()
    elif arg == "pos_weight":
        main_pos_weight()
    elif arg == "pos_weight_v2":
        main_pos_weight_v2()
    elif arg == "best_recheck":
        main_best_recheck()
    elif arg == "true_kfold":
        main_true_kfold()
    elif arg == "concept_proj":
        main_concept_proj()
    elif arg == "ext_val":
        main_ext_val()
    else:
        main()

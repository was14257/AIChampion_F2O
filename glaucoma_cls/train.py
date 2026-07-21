import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from torch.utils.data import DataLoader

from glaucoma_cls.data import build_frames, loaders, class_weights, FundusDS
from glaucoma_cls.model import GlaucomaNet
from local_config import (CLS_EPOCHS, CLS_BATCH_SIZE, CLS_LR_ENC, CLS_LR_HEAD,
                          CLS_FREEZE, CLS_ADD_GAMMA_TRAIN, CLS_PREDICT_ONLY)

OUT = Path("C:/Users/hogri/OneDrive/Desktop/AIGS자율공모/Code/outputs")
OUT = OUT if OUT.parent.exists() else Path("/home/tta/outputs")
OUT = OUT / "glaucoma_cls"
CKPT = OUT / "best.pth"


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
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
    return {"auc": auc, "acc": acc, "sens": sens, "spec": spec}, probs


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    ext, val, test = build_frames()
    print(f"학습 {len(ext)} / 검증(GAMMA train) {len(val)} / 타겟(GAMMA test) {len(test)}")

    model = GlaucomaNet(freeze_encoder=CLS_FREEZE).to(device)
    ngpu = torch.cuda.device_count()
    if ngpu > 1 and not CLS_PREDICT_ONLY:
        model = torch.nn.DataParallel(model)
    _m = model.module if isinstance(model, torch.nn.DataParallel) else model

    if not CLS_PREDICT_ONLY:
        if CLS_ADD_GAMMA_TRAIN:
            ext = pd.concat([ext, val[["path", "label", "dataset"]]], ignore_index=True)
            print(f"  GAMMA train 포함 → 학습 {len(ext)}")
        tr, va = loaders(ext, val, batch_size=CLS_BATCH_SIZE)
        pos_w = torch.tensor([class_weights(ext)], device=device)
        opt = torch.optim.AdamW(_m.param_groups(CLS_LR_ENC, CLS_LR_HEAD), weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CLS_EPOCHS)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

        best = -1
        for ep in range(1, CLS_EPOCHS + 1):
            model.train()
            t0 = time.time()
            tot = 0
            for x, y in tr:
                x = x.to(device)
                y = y.float().to(device)
                opt.zero_grad()
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_w)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item()
            sched.step()
            m, _ = evaluate(model, va, device)
            print(f"[{ep:2d}/{CLS_EPOCHS}] loss={tot/len(tr):.4f}  "
                  f"GAMMA-val AUC={m['auc']:.3f} acc={m['acc']:.3f} "
                  f"sens={m['sens']:.2f} spec={m['spec']:.2f} ({time.time()-t0:.0f}s)")
            if m["auc"] > best:
                best = m["auc"]
                torch.save({"model": _m.state_dict(), "val": m, "epoch": ep}, CKPT)
                print(f"   -> best 저장 (AUC={best:.3f})")
        print(f"\n학습 완료. best GAMMA-val AUC = {best:.3f}")

    ck = torch.load(CKPT, map_location=device, weights_only=False)
    _m.load_state_dict(ck["model"])
    _m.to(device)
    print(f"best 체크포인트 로드 (val AUC={ck['val']['auc']:.3f})")

    val_loader = DataLoader(FundusDS(val, train=False), batch_size=CLS_BATCH_SIZE, num_workers=4)
    vm, vprobs = evaluate(_m, val_loader, device)
    vlabels = val["label"].values
    fpr, tpr, thr = roc_curve(vlabels, vprobs)
    thr_opt = float(thr[np.argmax(tpr - fpr)])
    vpred = (vprobs >= thr_opt).astype(int)
    tn, fp, fn, tp = confusion_matrix(vlabels, vpred, labels=[0, 1]).ravel()
    print(f"보정 threshold={thr_opt:.3f} | GAMMA-val(보정후) "
          f"acc={(tp+tn)/len(vlabels):.3f} sens={tp/max(1,tp+fn):.2f} spec={tn/max(1,tn+fp):.2f}")

    test_loader = DataLoader(FundusDS(test, train=False, with_label=False),
                             batch_size=CLS_BATCH_SIZE, num_workers=4)
    _m.eval()
    rows = []
    with torch.no_grad():
        for x, cid in test_loader:
            x = x.to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                p = torch.sigmoid(_m(x)).float().cpu().numpy()
            for c, pr in zip(cid, p):
                rows.append({"case_id": c, "glaucoma_prob": round(float(pr), 4),
                             "pseudo_label": int(pr >= thr_opt)})
    df = pd.DataFrame(rows)
    out_csv = OUT / "gamma_test_pseudolabels.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\npseudo-label 저장: {out_csv}")
    print("분포:", df["pseudo_label"].value_counts().to_dict(),
          "| 애매(0.4~0.6):", int(df["glaucoma_prob"].between(0.4, 0.6).sum()))


if __name__ == "__main__":
    main()

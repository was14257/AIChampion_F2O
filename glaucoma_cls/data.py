from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

_LOCAL = Path("C:/Users/hogri/OneDrive/Desktop/AIGS자율공모/Code/data")
_SERVER = Path("/home/tta/data")
DATA = _LOCAL if _LOCAL.exists() else _SERVER
_GD = Path("D:/GAMMA")
GAMMA = _GD if _GD.exists() else DATA / "GAMMA"
GMM = GAMMA / "grading/Glaucoma_grading"

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _refuge():
    d = DATA / "REFUGE/train/Images"
    return [(str(p), 1 if p.stem.lower().startswith("g") else 0, "REFUGE")
            for p in sorted(d.glob("*.jpg"))]


def _origa():
    df = pd.read_csv(DATA / "ORIGA/OrigaList.csv")
    d = DATA / "ORIGA/Images"
    out = []
    for _, r in df.iterrows():
        p = d / str(r["Filename"])
        if p.exists():
            out.append((str(p), int(r["Glaucoma"]), "ORIGA"))
    return out


def _g1020():
    df = pd.read_csv(DATA / "G1020/G1020.csv")
    d = DATA / "G1020/Images"
    out = []
    for _, r in df.iterrows():
        p = d / str(r["imageID"])
        if p.exists():
            out.append((str(p), int(r["binaryLabels"]), "G1020"))
    return out


def _gamma_train():
    df = pd.read_excel(GMM / "training/glaucoma_grading_training_GT.xlsx")
    out = []
    for _, r in df.iterrows():
        cid = f"{int(r['data']):04d}"
        p = GMM / f"training/multi-modality_images/{cid}/{cid}.jpg"
        if p.exists():
            lab = 0 if int(r["non"]) == 1 else 1
            out.append((str(p), lab, "GAMMA_train", cid))
    return out


def _gamma_test():
    d = GMM / "testing/multi-modality_images"
    out = []
    for cd in sorted(d.iterdir()):
        if cd.is_dir():
            p = cd / f"{cd.name}.jpg"
            if p.exists():
                out.append((str(p), cd.name))
    return out


def build_frames():
    ext = pd.DataFrame(_refuge() + _origa() + _g1020(),
                       columns=["path", "label", "dataset"])
    val = pd.DataFrame(_gamma_train(), columns=["path", "label", "dataset", "case_id"])
    test = pd.DataFrame(_gamma_test(), columns=["path", "case_id"])
    return ext, val, test


class FundusDS(Dataset):

    def __init__(self, df, img_size=224, train=False, with_label=True):
        self.df = df.reset_index(drop=True)
        self.with_label = with_label
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(0.2, 0.2, 0.1),
                transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = self.tf(Image.open(r["path"]).convert("RGB"))
        if self.with_label:
            return img, int(r["label"])
        return img, r.get("case_id", str(i))


def class_weights(df):
    n = (df["label"] == 0).sum()
    p = (df["label"] == 1).sum()
    return float(n / max(1, p))


def loaders(ext, val, batch_size=64, img_size=224, num_workers=8):
    kw = dict(num_workers=num_workers, pin_memory=True)
    tr = DataLoader(FundusDS(ext, img_size, train=True), batch_size=batch_size,
                    shuffle=True, drop_last=True, **kw)
    va = DataLoader(FundusDS(val, img_size, train=False), batch_size=batch_size,
                    shuffle=False, **kw)
    return tr, va


if __name__ == "__main__":
    ext, val, test = build_frames()
    print("외부(학습):", len(ext), ext["dataset"].value_counts().to_dict())
    print("  라벨:", ext["label"].value_counts().to_dict(), "| pos_weight=%.2f" % class_weights(ext))
    print("GAMMA train(검증):", len(val), val["label"].value_counts().to_dict())
    print("GAMMA test(타겟):", len(test))

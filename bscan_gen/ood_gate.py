import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from bscan_gen.utils import load_retfound_encoder, embed_fundus
from local_config import GATE_CHECK_IMAGE

OUTF = CFG.paths.oct_features
GATE = OUTF / "ood_gate.npz"
_ENC = None


def _encoder():
    """Lazily load and cache the RETFound encoder."""
    global _ENC
    if _ENC is None:
        _ENC = load_retfound_encoder()
    return _ENC


def embed(im):
    """Embed a fundus image using the cached RETFound encoder."""
    return embed_fundus(_encoder(), im)


def build(n_pca=30, pct=99):
    """Fit a PCA + Mahalanobis-distance out-of-distribution gate on in-distribution embeddings."""
    Xk = np.load(OUTF / "emb_whole_final.npz", allow_pickle=True)["X"]
    sc = StandardScaler().fit(Xk)
    pca = PCA(n_pca, random_state=0).fit(sc.transform(Xk))
    Z = pca.transform(sc.transform(Xk))
    mu = Z.mean(0)
    cov = np.cov(Z.T)
    icov = np.linalg.inv(cov + 1e-3 * np.eye(n_pca))
    d = np.array([np.sqrt((z - mu) @ icov @ (z - mu)) for z in Z])
    thr = float(np.percentile(d, pct))
    np.savez(
        GATE, mean_=sc.mean_, scale_=sc.scale_, comps=pca.components_,
        pca_mean=pca.mean_, mu=mu, icov=icov, thr=thr,
    )
    print(f"gate saved -> {GATE}\n  in-dist distance median={np.median(d):.1f} p{pct}=thr={thr:.1f}")


class Gate:
    """Loads a saved OOD gate and scores/checks new fundus embeddings against it."""

    def __init__(self):
        if not GATE.exists():
            raise FileNotFoundError("Run `python ood_gate.py build` first")
        self.z = np.load(GATE)
        self.thr = float(self.z["thr"])

    def score(self, e):
        """Mahalanobis distance of embedding e from the in-distribution PCA space."""
        x = (e - self.z["mean_"]) / self.z["scale_"]
        zp = (x - self.z["pca_mean"]) @ self.z["comps"].T
        d = zp - self.z["mu"]
        return float(np.sqrt(d @ self.z["icov"] @ d))

    def check_image(self, im):
        """Return (is_in_distribution, distance) for a fundus image."""
        d = self.score(embed(im))
        return d <= self.thr, d


if __name__ == "__main__":
    if GATE_CHECK_IMAGE is None:
        build()
    else:
        g = Gate()
        ok, d = g.check_image(Image.open(GATE_CHECK_IMAGE))
        verdict = "OK fundus passed" if ok else "REJECT (non-fundus)"
        print(f"distance={d:.1f}  thr={g.thr:.1f}  ->  {verdict}")

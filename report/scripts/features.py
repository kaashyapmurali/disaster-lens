"""Encoder feature images for the main explorer scene (optional, heavy: pretrained DINOv3 ViT-L on CPU or Apple GPU).

    uv run --group vit --group notebook python report/scripts/features.py

Writes report/bundle/images/features-{pre,post,change}.jpg (880 px): the first three principal components of the
patch features per date, and Δ features = 1 − cosine similarity between each patch's pre and post features. These
illustrate the damage model's inputs; they are not model outputs. Needs the local DINOv3 weights (see README).
"""

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from paths import DATASETS, DINOV3, IMAGES
from PIL import Image
from scenes import MAIN_CROP, MAIN_SCENE

from disaster_lens.damage.dinov3 import SATELLITE_MEAN, SATELLITE_STD, DINOv3Backbone
from disaster_lens.detection.verifier import load_image


def save(rgb: torch.Tensor, name: str) -> None:
    """[3, 64, 64] in [0, 1] → upsampled to the 1024 px scene, cropped like the explorer, saved as JPEG."""
    up = F.interpolate(rgb[None], size=(1024, 1024), mode="bicubic", align_corners=False)[0].clamp(0, 1)
    image = Image.fromarray((up.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).crop(MAIN_CROP)
    image.save(IMAGES / name, quality=88, optimize=True)


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder = DINOv3Backbone.from_local_checkpoint(DINOV3).to(device).eval()
    maps = {}
    for phase in ("pre", "post"):
        image = load_image(DATASETS, f"raw/test/images/{MAIN_SCENE}_{phase}_disaster.png", SATELLITE_MEAN, SATELLITE_STD)
        with torch.inference_mode():
            maps[phase] = encoder(image.to(device)[None]).spatial_map[0].float().cpu()
    for phase, features in maps.items():
        tokens = features.flatten(1).T
        mean = tokens.mean(0)
        _, _, v = torch.pca_lowrank(tokens - mean, q=3, center=False)
        projected = (tokens - mean) @ v
        lo, hi = torch.quantile(projected, .02, dim=0), torch.quantile(projected, .98, dim=0)
        save(((projected - lo) / (hi - lo)).T.reshape(3, 64, 64), f"features-{phase}.jpg")
    change = 1 - F.cosine_similarity(maps["pre"], maps["post"], dim=0)
    lo, hi = torch.quantile(change, .02), torch.quantile(change, .98)
    scaled = ((change - lo) / (hi - lo)).clamp(0, 1).numpy()
    save(torch.from_numpy(colormaps["magma"](scaled)[..., :3]).permute(2, 0, 1).float(), "features-change.jpg")
    print(f"features: Δ features range {float(change.min()):.3f}–{float(change.max()):.3f} on {device}")


if __name__ == "__main__":
    main()

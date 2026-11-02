"""Run the reported fine-tuned DINOv3 damage model on prepared Mexico Beach tiles (conditions A, B, D), Modal L4.

    modal volume put disaster-lens-data outputs/drone/inputs/mexico-beach-damage /alignment/mexico-beach-damage --env main --profile kaashyapmurali
    modal run --detach --env main --profile kaashyapmurali drone/modal_damage.py --run-name damage-mb-01
    modal volume get disaster-lens-results /alignment/damage-mb-01 outputs/drone/runs/ --env main --profile kaashyapmurali

Uses the project's own environment (same image recipe as modal_app.py) and the checkpoint behind the report:
/runs/vit-finetune-bf16-l4-20261020-01/best.pt. Output: predictions.csv (day, condition, building_id, label, probs).
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/root")  # project root, only needed locally
app = modal.App("disaster-lens-damage-check", include_source=False)
image = (modal.Image.debian_slim(python_version="3.14.8").apt_install("libgomp1")
         .uv_sync(str(ROOT), frozen=True, uv_version="0.12.24", groups=["vit"])
         .env({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"})
         .add_local_python_source("disaster_lens")
         .add_local_file(Path(__file__).resolve(), "/root/modal_damage.py"))
data_volume = modal.Volume.from_name("disaster-lens-data", version=1)
results_volume = modal.Volume.from_name("disaster-lens-results", version=1)
DEFAULT_RUN = "vit-finetune-bf16-l4-20261020-01"  # the model behind the report


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(16384, 16384), timeout=1800,
              volumes={"/data": data_volume.with_mount_options(read_only=True), "/runs": results_volume})
def predict(run_name: str, checkpoint_run: str = DEFAULT_RUN) -> dict:
    import csv

    import numpy as np
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image

    from disaster_lens.damage.classifier import normalization
    from disaster_lens.damage.predictors import Predictor
    from disaster_lens.damage.train_live import load_live_model
    from disaster_lens.data.manifest import DAMAGE_LABELS

    out = Path("/runs/alignment") / run_name
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new run name")
    out.mkdir(parents=True)
    device = torch.device("cuda")
    checkpoint = torch.load(f"/runs/{checkpoint_run}/best.pt", map_location="cpu", weights_only=True)
    model, recipe = load_live_model(checkpoint, device)
    predictor = Predictor(model, recipe["precision"])
    mean, std = normalization(recipe["encoder"])
    size = recipe["image_size"]

    def tensor(arr):
        img = Image.fromarray(arr).convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        return TF.normalize(TF.to_tensor(img), mean, std)[None].to(device)

    rows, counts = [], {}
    for path in sorted(Path("/data/alignment/mexico-beach-damage").glob("tiles_day*.npz")):
        z = np.load(path)
        day = int(path.stem[-1])
        scale = size / z["pre"].shape[1]
        for t in range(len(z["pre"])):
            sel = np.where(z["tile"] == t)[0]
            if len(sel) == 0:
                continue
            boxes = torch.from_numpy(z["boxes"][sel] * scale).float().to(device)
            pre = tensor(z["pre"][t])
            for cond in ("A", "B", "D"):
                probs = predictor.predict(pre, tensor(z[cond][t]), {"b": (boxes, boxes)}, device)["b"].numpy()
                for j, p in zip(sel, probs):
                    rows.append([day, cond, z["building_id"][j], z["label"][j], *[f"{v:.6f}" for v in p]])
                counts[cond] = counts.get(cond, 0) + len(sel)
        print(f"{path.name}: done", flush=True)
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["day", "condition", "building_id", "label", *[f"p_{c}" for c in DAMAGE_LABELS]])
        w.writerows(rows)
    results_volume.commit()
    return {"rows": len(rows), "per_condition": counts, "image_size": size, "precision": recipe["precision"],
            "checkpoint_run": checkpoint_run, "epoch": checkpoint["epoch"]}


@app.local_entrypoint()
def main(run_name: str, checkpoint_run: str = DEFAULT_RUN):
    print(predict.remote(run_name, checkpoint_run))

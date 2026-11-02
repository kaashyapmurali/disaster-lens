"""Learned matching (LightGlue+DISK, RoMa) for drone -> satellite alignment, on a Modal L4 GPU.

    modal volume put disaster-lens-data outputs/drone/inputs/mexico-beach /alignment/mexico-beach --env main --profile kaashyapmurali
    modal run --detach drone/modal_match.py --run-name align-mb-01 --env main --profile kaashyapmurali
    modal volume get disaster-lens-results /alignment/align-mb-01 outputs/drone/runs/ --env main --profile kaashyapmurali

Inputs: /data/alignment/<site>/*.npz from `python -m disaster_lens.drone.inputs alignment` (ref, drone, masks, 0.5 m grid). Each reference tile is
matched against the same drone area plus a PAD margin (the displacement prior); matches outside either mask or with
displacement > PRIOR_PX are dropped. Output per input and method: /runs/alignment/<run>/<input>_<method>.npz with
global pixel coordinates a (reference), b (drone) and a score.
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("disaster-lens-align", include_source=False)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libgl1", "libglib2.0-0", "git")
         .pip_install("torch==2.5.1", "torchvision==0.20.1", "kornia==0.8.0", "romatch",
                      "opencv-python-headless", "numpy<2", "pillow")
         .env({"PYTHONUNBUFFERED": "1"})
         .add_local_file(Path(__file__).resolve(), "/root/modal_match.py"))  # include_source=False: ship this file
data_volume = modal.Volume.from_name("disaster-lens-data", version=1)
results_volume = modal.Volume.from_name("disaster-lens-results", version=1)

TILE, STEP, PAD = 768, 640, 48  # px at 0.5 m: 384 m tiles, 320 m step, 24 m search margin
PRIOR_PX = 40  # 20 m


@app.function(image=image, gpu="L4", timeout=3600, cpu=4, memory=16384,
              volumes={"/data": data_volume.with_mount_options(read_only=True), "/runs": results_volume})
def match(run_name: str, site: str, methods: list[str]) -> dict:
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    out = Path("/runs/alignment") / run_name
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new run name")
    out.mkdir(parents=True)
    torch.set_grad_enabled(False)  # inference only
    dev = torch.device("cuda")
    t_start = time.time(); report = {}

    matchers = {}
    if "lightglue" in methods:
        import kornia.feature as KF
        disk = KF.DISK.from_pretrained("depth").to(dev).eval()
        lg = KF.LightGlueMatcher("disk").to(dev).eval()

        def lightglue(ra, da):
            def feats(img):
                t = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(dev)
                f = disk(t, n=4096, window_size=5, score_threshold=0.0, pad_if_not_divisible=True)[0]
                return f.keypoints, f.descriptors
            k1, d1 = feats(ra); k2, d2 = feats(da)
            if len(k1) < 8 or len(k2) < 8:
                return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
            lafs = lambda k: KF.laf_from_center_scale_ori(k[None], torch.ones(1, len(k), 1, 1, device=dev))
            dist, idx = lg(d1, d2, lafs(k1), lafs(k2), hw1=torch.tensor(ra.shape[:2]), hw2=torch.tensor(da.shape[:2]))
            idx = idx.cpu().numpy()
            return k1[idx[:, 0]].cpu().numpy(), k2[idx[:, 1]].cpu().numpy(), 1 - dist.squeeze(-1).cpu().numpy()
        matchers["lightglue"] = lightglue

    if "roma" in methods:
        from romatch import roma_outdoor
        roma = roma_outdoor(device=dev, coarse_res=560, upsample_res=864, use_custom_corr=False)  # no compiled local_corr kernel

        def roma_match(ra, da):
            warp, cert = roma.match(Image.fromarray(ra), Image.fromarray(da), device=dev)
            m, c = roma.sample(warp, cert, num=3000)
            ka, kb = roma.to_pixel_coordinates(m, ra.shape[0], ra.shape[1], da.shape[0], da.shape[1])
            return ka.cpu().numpy(), kb.cpu().numpy(), c.cpu().numpy()
        matchers["roma"] = roma_match

    for path in sorted(Path(f"/data/alignment/{site}").glob("*.npz")):
        z = np.load(path)
        ref, drone, m_ref, m_drone = z["ref"], z["drone"], z["m_ref"], z["m_drone"]
        H, W = m_ref.shape
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm = lambda im: cv2.cvtColor(clahe.apply(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)), cv2.COLOR_GRAY2RGB)
        ref_n, drone_n = norm(ref), norm(drone)
        for name, fn in matchers.items():
            t0 = time.time(); A, B, S = [], [], []; tiles = 0
            for y in range(0, max(1, H - TILE + STEP), STEP):
                for x in range(0, max(1, W - TILE + STEP), STEP):
                    y1, x1 = min(y + TILE, H), min(x + TILE, W)
                    if m_ref[y:y1, x:x1].mean() < 0.05:
                        continue
                    dy0, dx0 = max(0, y - PAD), max(0, x - PAD)
                    dy1, dx1 = min(H, y1 + PAD), min(W, x1 + PAD)
                    a, b, s = fn(ref_n[y:y1, x:x1], drone_n[dy0:dy1, dx0:dx1])
                    tiles += 1
                    if len(a) == 0:
                        continue
                    a = a + [x, y]; b = b + [dx0, dy0]
                    ai, bi = np.round(a).astype(int), np.round(b).astype(int)
                    ok = ((ai[:, 0] >= 0) & (ai[:, 0] < W) & (ai[:, 1] >= 0) & (ai[:, 1] < H) &
                          (bi[:, 0] >= 0) & (bi[:, 0] < W) & (bi[:, 1] >= 0) & (bi[:, 1] < H))
                    a, b, s, ai, bi = a[ok], b[ok], s[ok], ai[ok], bi[ok]
                    ok = m_ref[ai[:, 1], ai[:, 0]] & m_drone[bi[:, 1], bi[:, 0]] & (np.hypot(*(a - b).T) < PRIOR_PX)
                    A.append(a[ok]); B.append(b[ok]); S.append(s[ok])
            A = np.concatenate(A) if A else np.zeros((0, 2)); B = np.concatenate(B) if B else np.zeros((0, 2))
            S = np.concatenate(S) if S else np.zeros(0)
            np.savez_compressed(out / f"{path.stem}_{name}.npz", a=A, b=B, score=S)
            report[f"{path.stem}_{name}"] = {"tiles": tiles, "matches": int(len(A)), "seconds": round(time.time() - t0, 1)}
            print(f"{path.stem} {name}: {tiles} tiles, {len(A)} matches kept, {time.time() - t0:.0f}s", flush=True)
    report["total_seconds"] = round(time.time() - t_start, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    results_volume.commit()
    return report


@app.local_entrypoint()
def main(run_name: str, site: str = "mexico-beach", methods: str = "lightglue,roma"):
    print(match.remote(run_name, site, methods.split(",")))

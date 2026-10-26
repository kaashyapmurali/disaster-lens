"""Run the DisasterLens pipeline stages on a Modal L4 GPU.

Each workflow is a local entrypoint that builds one or more `python -m disaster_lens...` commands and submits them
to a single remote runner:

    modal run --detach modal_app.py::damage_features --run-name resnet18-features --encoder resnet18 --confirm-run
    modal run --detach modal_app.py::damage_head     --run-name resnet18-paired   --cache resnet18-features --confirm-run
    modal run --detach modal_app.py::damage_live     --run-name vit-finetuned --encoder dinov3-vitl --inputs paired \
        --trainable-blocks 4 --init-head vit-frozen --precision bf16 --confirm-run
    modal run --detach modal_app.py::detector_train  --run-name maskrcnn --weights-sha256 SHA --confirm-run
    modal run --detach modal_app.py::detector_crops  --run-name maskrcnn-2x --parent maskrcnn --confirm-run
    modal run --detach modal_app.py::detector_export --run maskrcnn-2x --tile-size 512 --confirm-run
    modal run --detach modal_app.py::end_to_end      --run-name e2e --damage-run vit-finetuned --detector-run maskrcnn-2x --confirm-run

Always use `modal run --detach`: runs are spawned, and without --detach Modal stops the app (cancelling the run)
as soon as the local entrypoint returns.

Volumes: `disaster-lens-data` (read-only inputs: datasets/processed and datasets/raw, uploaded with
`modal volume put`) and `disaster-lens-results` (runs, feature caches, pretrained weights under models/).
The runner never overwrites an existing run, enforces a work time limit by killing the stage's process group,
writes `<run>.status.json` and `<run>.log` next to the run, and never retries. `--confirm-run` is required.
"""

import json
import shlex
from collections.abc import Callable
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
DATA, RUNS = Path("/data"), Path("/runs")
SPLITS = DATA / "processed/splits/seed42"
INDEX = DATA / "processed/localization/seed42-v1"
TEST_INDEX = DATA / "processed/localization/test-v1"
TEST_MANIFEST = DATA / "processed/test_buildings.csv"
DINOV3 = RUNS / "models/dinov3-vitl16-sat-timm-ce148bd5"
COCO = RUNS / "models/maskrcnn-resnet50-fpn-coco-v1/weights.pt"
CACHES = RUNS / "feature-caches"
MAX_SECONDS = 6 * 3600
RESERVED = ("models", "feature-caches")

app = modal.App("disaster-lens", include_source=False)
image = (modal.Image.debian_slim(python_version="3.14.8").apt_install("libgomp1")
         .uv_sync(str(ROOT), frozen=True, uv_version="0.12.24", groups=["vit"])
         .env({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"})
         .add_local_python_source("disaster_lens")
         .add_local_file(ROOT / "modal_app.py", "/root/modal_app.py"))   # the runner itself (include_source=False)
data_volume = modal.Volume.from_name("disaster-lens-data", version=1)
results_volume = modal.Volume.from_name("disaster-lens-results", create_if_missing=True, version=1)


def validate_name(name: str) -> str:
    if (not name or not name.isascii() or not all(c.isalnum() or c in "-_" for c in name) or name in RESERVED):
        raise ValueError("Use a non-reserved run name made of letters, numbers, '-' and '_'.")
    return name


def module(name: str, *args: str) -> list[str]:
    return ["python", "-u", "-m", f"disaster_lens.{name}", *args]


# ---------------------------------------------------------------- command builders (pure; tested)

def damage_features_stages(run_name: str, encoder: str) -> list[list[str]]:
    """Frozen-encoder features for both splits, cached under feature-caches/<run_name>."""
    validate_name(run_name)
    extra = ["--encoder-dir", str(DINOV3)] if encoder == "dinov3-vitl" else []
    batch = "1" if encoder == "dinov3-vitl" else "8"
    return [module("damage.feature_cache", "--data-dir", str(DATA), "--split-dir", str(SPLITS), "--cache-dir",
                   str(CACHES / run_name), "--encoder", encoder, *extra, "--splits", "train", "validation",
                   "--image-size", "1024", "--roi-size", "3", "--batch-size", batch, "--num-workers", "2", "--device", "cuda")]


def damage_head_stages(run_name: str, cache: str, epochs: int = 5) -> list[list[str]]:
    """Train the paired fusion head on cached features, then export validation predictions."""
    validate_name(run_name)
    validate_name(cache)
    output = RUNS / run_name
    return [module("damage.train_cached", "--train-cache", str(CACHES / cache / "train"), "--validation-cache",
                   str(CACHES / cache / "validation"), "--output-dir", str(output), "--epochs", str(epochs),
                   "--batch-size", "8", "--validation-batch-size", "16", "--lr", "0.001", "--weight-decay", "0.0001",
                   "--class-weight", "none", "--num-workers", "2", "--seed", "42", "--device", "cuda"),
            module("damage.export_cached", "--run-dir", str(output), "--validation-cache", str(CACHES / cache / "validation"),
                   "--batch-size", "16", "--num-workers", "2", "--device", "cuda")]


def damage_live_stages(run_name: str, encoder: str, inputs: str, trainable_blocks: int, *, init_head: str = "",
                       epochs: int = 5, precision: str = "float32", backbone_lr: float = 1e-5,
                       head_lr: float = 1e-4) -> list[list[str]]:
    """Train any encoder × inputs × trainable-blocks classifier on live images, then export validation predictions."""
    validate_name(run_name)
    output = RUNS / run_name
    extra = ["--encoder-dir", str(DINOV3)] if encoder == "dinov3-vitl" else []
    if init_head:
        extra += ["--init-head", str(RUNS / validate_name(init_head) / "best.pt")]
    return [module("damage.train_live", "train", "--encoder", encoder, "--inputs", inputs, "--trainable-blocks",
                   str(trainable_blocks), *extra, "--data-dir", str(DATA), "--split-dir", str(SPLITS), "--output-dir",
                   str(output), "--epochs", str(epochs), "--batch-size", "1", "--validation-batch-size", "1",
                   "--backbone-lr", repr(backbone_lr), "--head-lr", repr(head_lr), "--weight-decay", "0.0001",
                   "--precision", precision, "--num-workers", "2", "--seed", "42", "--device", "cuda"),
            module("damage.train_live", "export", "--run-dir", str(output), "--data-dir", str(DATA), "--manifest",
                   str(SPLITS / "validation.csv"), "--num-workers", "2", "--device", "cuda")]


DETECTOR_RECIPE = ["--data-dir", str(DATA), "--index-dir", str(INDEX), "--batch-size", "1", "--validation-batch-size", "1",
                   "--image-size", "1024", "--anchor-sizes", "16", "32", "64", "128", "256", "--trainable-backbone-layers",
                   "3", "--pre-nms-proposals", "4000", "--post-nms-proposals", "3000", "--max-detections", "2500",
                   "--score-floor", "0.001", "--operating-threshold", "0.5", "--backbone-lr", "0.001", "--head-lr",
                   "0.005", "--momentum", "0.9", "--weight-decay", "0.0005", "--geometry-policy", "make-valid-polygonal",
                   "--mask-threshold", "0.5", "--num-workers", "2", "--seed", "42", "--device", "cuda"]


def detector_train_stages(run_name: str, weights_sha256: str, epochs: int = 5) -> list[list[str]]:
    """Mask R-CNN on whole scenes at native scale, from COCO weights."""
    validate_name(run_name)
    return [module("detection.train", "train", "--fresh-training", "--output-dir", str(RUNS / run_name),
                   "--coco-weights", str(COCO), "--weights-sha256", weights_sha256, "--epochs", str(epochs), *DETECTOR_RECIPE)]


def detector_crops_stages(run_name: str, parent: str, *, epochs: int = 1, crops_per_scene: int = 4,
                          checkpoint: str = "best", validation_subset: int = 100) -> list[list[str]]:
    """Continue a detector on random 512 px crops upsampled 2×, validating with 2× tiles on a scene subset."""
    validate_name(run_name)
    return [module("detection.train", "train", "--resume-from", str(RUNS / validate_name(parent)), "--resume-checkpoint",
                   checkpoint, "--additional-epochs", str(epochs), "--allow-protocol-change", "--output-dir",
                   str(RUNS / run_name), *DETECTOR_RECIPE, "--crop-size", "512", "--crops-per-scene", str(crops_per_scene),
                   "--crop-min-visible", "0.5", "--crop-resample-empty", "5", "--validation-subset", str(validation_subset),
                   "--validation-tile-size", "512", "--validation-tile-min-overlap", "128", "--validation-tile-batch-size", "3")]


def detector_export_stages(run: str, *, checkpoint: str = "best", tile_size: int = 0, split: str = "validation") -> list[list[str]]:
    """Predictions (masks) of a trained detector on the validation or test split; tile_size > 0 uses upsampled tiles."""
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test.")
    tiles = ["--tile-size", str(tile_size), "--tile-min-overlap", "128", "--tile-batch-size", "3"] if tile_size else []
    return [module("detection.train", "export", "--run-dir", str(RUNS / validate_name(run)), "--checkpoint-name",
                   checkpoint, "--split", split, "--data-dir", str(DATA), "--index-dir", str(INDEX if split == "validation" else TEST_INDEX),
                   "--batch-size", "1", "--num-workers", "2", "--device", "cuda", *tiles)]


def end_to_end_stages(run_name: str, damage_run: str, detector_run: str, *, split: str = "validation",
                      detector_export: str = "", damage_checkpoint: str = "best",
                      saved_predictions: str = "validation_predictions/predictions.csv",
                      cached_dinov3_head: bool = False, no_change: bool = False) -> list[list[str]]:
    """Damage predictions on detected and annotated footprints. On validation the damage model must reproduce its
    recorded results; the test split has none, so that check is skipped there."""
    validate_name(run_name)
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test.")
    damage = RUNS / validate_name(damage_run)
    detector_export = detector_export or f"{split}_predictions_tile512"
    extra = ["--encoder-dir", str(DINOV3)] if cached_dinov3_head else []
    if split == "test" or no_change:
        extra += ["--skip-reference-check"] + (["--no-change"] if no_change else [])
    elif saved_predictions:
        extra += ["--saved-predictions", str(damage / saved_predictions)]
    manifest = SPLITS / "validation.csv" if split == "validation" else TEST_MANIFEST
    return [module("end_to_end.run", "--checkpoint", str(damage / f"{damage_checkpoint}.pt"), *extra,
                   "--manifest", str(manifest), "--data-dir", str(DATA), "--export-dir",
                   str(RUNS / validate_name(detector_run) / detector_export), "--output-dir", str(RUNS / run_name),
                   "--num-workers", "2", "--device", "cuda")]


# ---------------------------------------------------------------- remote runner

def verifier_features_stages(run_name: str, detector_run: str, *, split: str = "validation",
                             floor: float = .1) -> list[list[str]]:
    """Frozen DINOv3 features (2x tiles) for every detector candidate >= floor, for the detection verifier."""
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test.")
    export = RUNS / validate_name(detector_run) / f"{split}_predictions_tile512"
    return [module("detection.verifier", "extract", "--export-dir", str(export), "--data-dir", str(DATA),
                   "--encoder-dir", str(DINOV3), "--output-dir", str(RUNS / validate_name(run_name)),
                   "--floor", str(floor), "--device", "cuda")]


def run_stages(run_name: str, stages: list[list[str]], work_seconds: int, *, runs: Path = RUNS,
               commit: Callable[[], None] = lambda: None) -> dict:
    """Run stages in order with one shared work budget. Refuses existing outputs; never retries."""
    import os
    import signal
    import subprocess
    import threading
    import time

    validate_name(run_name)
    if not 0 < work_seconds <= MAX_SECONDS:
        raise ValueError(f"work_seconds must be in (0, {MAX_SECONDS}].")
    status_path, log_path = runs / f"{run_name}.status.json", runs / f"{run_name}.log"
    if (runs / run_name).exists() or status_path.exists() or log_path.exists():
        raise FileExistsError(f"{run_name} already exists; inspect it rather than overwrite or retry.")
    started = time.monotonic()
    status = {"state": "running", "stages": [shlex.join(stage) for stage in stages], "completed_stages": 0,
              "work_seconds": work_seconds}

    lock = threading.Lock()

    def persist() -> None:
        with lock:
            status["elapsed_seconds"] = round(time.monotonic() - started, 2)
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            commit()

    stop_heartbeat = threading.Event()

    def heartbeat() -> None:           # keep status current during quiet stretches of a stage
        while not stop_heartbeat.wait(60):
            persist()

    with status_path.open("x") as stream:
        stream.write(json.dumps(status, indent=2) + "\n")
    environment = {**os.environ, "TORCH_HOME": str(runs / ".torch-cache")}
    process = None
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        with log_path.open("x") as log:
            for index, command in enumerate(stages, start=1):
                remaining = work_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Work budget exhausted before the next stage.")
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                           env=environment, start_new_session=True)
                expired = threading.Event()

                def kill(target=process, flag=expired) -> None:
                    flag.set()
                    if target.poll() is None:
                        os.killpg(target.pid, signal.SIGKILL)

                timer = threading.Timer(remaining, kill)
                timer.daemon = True
                timer.start()
                last_persist = 0.
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                    if time.monotonic() - last_persist > 30:
                        status["progress"] = line.strip()[:300]
                        persist()
                        last_persist = time.monotonic()
                code = process.wait()
                process.stdout.close()
                timer.cancel()
                if expired.is_set():
                    raise TimeoutError(f"Stage {index} exceeded the work budget and was stopped.")
                if code:
                    raise RuntimeError(f"Stage {index} exited with status {code}; see {log_path.name}. No retry.")
                status["completed_stages"] = index
                persist()
        status["state"] = "complete"
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stop_heartbeat.set()
        if process is not None and not process.stdout.closed:
            process.stdout.close()
        persist()
    return status


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(8192, 8192), timeout=MAX_SECONDS + 600, startup_timeout=600,
              retries=0, max_containers=1, scaledown_window=2, single_use_containers=True,
              volumes={str(DATA): data_volume.with_mount_options(read_only=True), str(RUNS): results_volume})
def run(run_name: str, stages: list[list[str]], work_seconds: int) -> dict:
    return run_stages(run_name, stages, work_seconds, commit=results_volume.commit)


# ---------------------------------------------------------------- local entrypoints

def submit(record_name: str, stages: list[list[str]], work_seconds: int, confirm_run: bool) -> None:
    """Print the plan; with --confirm-run, record and spawn it (the local record prevents accidental relaunch)."""
    validate_name(record_name)
    print(json.dumps({"run": record_name, "work_seconds": work_seconds, "stages": [shlex.join(s) for s in stages]}, indent=2))
    if not confirm_run:
        print("Dry run. Add --confirm-run to start it on an L4.")
        return
    record = ROOT / "outputs/modal" / f"{record_name}.json"
    if record.exists():
        raise FileExistsError(f"{record} exists; this run was already submitted.")
    record.parent.mkdir(parents=True, exist_ok=True)
    call = run.spawn(record_name, stages, work_seconds)
    record.write_text(json.dumps({"run": record_name, "call_id": call.object_id, "work_seconds": work_seconds,
                                  "stages": [shlex.join(s) for s in stages]}, indent=2) + "\n")
    print(f"Submitted {record_name}: {call.object_id} (requires `modal run --detach`; otherwise the run is cancelled)")


@app.local_entrypoint()
def damage_features(run_name: str, encoder: str, work_seconds: int = 5400, confirm_run: bool = False) -> None:
    submit(run_name, damage_features_stages(run_name, encoder), work_seconds, confirm_run)


@app.local_entrypoint()
def damage_head(run_name: str, cache: str, epochs: int = 5, work_seconds: int = 3000, confirm_run: bool = False) -> None:
    submit(run_name, damage_head_stages(run_name, cache, epochs), work_seconds, confirm_run)


@app.local_entrypoint()
def damage_live(run_name: str, encoder: str, inputs: str, trainable_blocks: int = 0, init_head: str = "",
                epochs: int = 5, precision: str = "float32", backbone_lr: float = 1e-5, head_lr: float = 1e-4,
                work_seconds: int = 21000, confirm_run: bool = False) -> None:
    stages = damage_live_stages(run_name, encoder, inputs, trainable_blocks, init_head=init_head, epochs=epochs,
                                precision=precision, backbone_lr=backbone_lr, head_lr=head_lr)
    submit(run_name, stages, work_seconds, confirm_run)


@app.local_entrypoint()
def detector_train(run_name: str, weights_sha256: str, epochs: int = 5, work_seconds: int = 14100,
                   confirm_run: bool = False) -> None:
    submit(run_name, detector_train_stages(run_name, weights_sha256, epochs), work_seconds, confirm_run)


@app.local_entrypoint()
def detector_crops(run_name: str, parent: str, epochs: int = 1, checkpoint: str = "best", work_seconds: int = 3900,
                   confirm_run: bool = False) -> None:
    stages = detector_crops_stages(run_name, parent, epochs=epochs, checkpoint=checkpoint)
    submit(run_name, stages, work_seconds, confirm_run)


@app.local_entrypoint()
def detector_export(run: str, checkpoint: str = "best", tile_size: int = 0, split: str = "validation",
                    work_seconds: int = 2400, confirm_run: bool = False) -> None:
    name = f"{run}-export-{split}-{checkpoint}" + (f"-tile{tile_size}" if tile_size else "")
    submit(name, detector_export_stages(run, checkpoint=checkpoint, tile_size=tile_size, split=split), work_seconds, confirm_run)


@app.local_entrypoint()
def end_to_end(run_name: str, damage_run: str, detector_run: str, split: str = "validation", detector_export: str = "",
               saved_predictions: str = "validation_predictions/predictions.csv", cached_dinov3_head: bool = False,
               no_change: bool = False, work_seconds: int = 1800, confirm_run: bool = False) -> None:
    stages = end_to_end_stages(run_name, damage_run, detector_run, split=split, detector_export=detector_export,
                               saved_predictions=saved_predictions, cached_dinov3_head=cached_dinov3_head,
                               no_change=no_change)
    submit(run_name, stages, work_seconds, confirm_run)


@app.local_entrypoint()
def verifier_features(run_name: str, detector_run: str, split: str = "validation", work_seconds: int = 2400,
                      confirm_run: bool = False) -> None:
    submit(run_name, verifier_features_stages(run_name, detector_run, split=split), work_seconds, confirm_run)

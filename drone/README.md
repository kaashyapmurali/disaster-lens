# Drone imagery (phase two)

Align a drone orthomosaic of a new disaster to a satellite pre-event reference with building footprints, then assess
damage per building with the DisasterLens model; measure both steps against CRASAR-U-DROIDs ground truth, and adapt the
model to drone imagery (fine-tuning on another hurricane).

- Library code: `src/disaster_lens/drone/` (`crasar`, `geo`, `alignment`, `inputs`, `evaluate`).
- Scripts here: Modal jobs (`modal_match.py`, `modal_damage.py`, `modal_crasar_prep.py`), the fine-tuning split
  (`make_split.py`) and preview sheets (`previews.py`).
- Dependencies: `uv sync --group drone` (rasterio, OpenCV, SciPy); Modal jobs also need `--group cloud`.
- Data (not in Git): `datasets/raw/drone/` (sources in `datasets/SOURCES.md`); intermediate files in
  `outputs/drone/{inputs,runs,previews}`.

Modal commands use `--env main --profile kaashyapmurali`, always `modal run --detach`; check `modal app list` after.

## Data

[CRASAR-U-DROIDs](https://huggingface.co/datasets/CRASAR/CRASAR-U-DROIDs) (CC BY 4.0): drone orthomosaics with
satellite and aircraft images of the same areas, Microsoft building footprints, damage labels on the xBD scale and, per
image, translation vectors that move each footprint onto its building (the alignment ground truth).

- **Test:** Mexico Beach, Hurricane Michael (`test/`): two drone mosaics (2 cm per pixel), Maxar pre-event images of
  2017-03-11 and 2018-07-28, a post-event image of 2018-10-12; 1,110 labelled buildings. Not seen in training.
- **Fine-tuning:** Hurricane Ian (`train/` and `valid/`): 24 drone mosaics with Maxar pre-event images, prepared on Modal
  (step 4); 23 used (Kennedy Green's pre-event image is panchromatic). The raw imagery is not stored.

## 1. Ground-truth error of the GPS placement

```bash
uv run --group drone python -m disaster_lens.drone.evaluate ground-truth
```

Per drone mosaic: footprint error under the drone's own georeference (median 1.8–5.7 m) and what one global translation
or affine leaves (the non-uniform part).

## 2. Alignment (Mexico Beach)

```bash
uv run --group drone python -m disaster_lens.drone.inputs alignment                 # 0.5 m pairs + masks
modal volume put disaster-lens-data outputs/drone/inputs/mexico-beach /alignment/mexico-beach
modal run --detach drone/modal_match.py --run-name align-mb-03                        # LightGlue + RoMa, L4
modal volume get disaster-lens-results /alignment/align-mb-03 outputs/drone/runs/
uv run --group drone python -m disaster_lens.drone.evaluate matches outputs/drone/runs/align-mb-03
```

Both images on the drone mosaic's 0.5 m UTM grid; buildings masked (reference footprints + 3 m, drone footprints at
their raw position + 10 m); matches implying more than 20 m dropped; RANSAC affine, then a Gaussian local correction
(25 m) from locally consistent matches. Result: median error 2.0–3.7 m (GPS) → 1.4–2.1 m (RoMa + local correction).

## 3. Damage on drone images (scenarios A, B, C)

```bash
uv run --group drone python -m disaster_lens.drone.inputs damage --matches outputs/drone/runs/align-mb-03
modal volume put disaster-lens-data outputs/drone/inputs/mexico-beach-damage /alignment/mexico-beach-damage
modal run --detach drone/modal_damage.py --run-name damage-mb-02                                     # xBD model
modal run --detach drone/modal_damage.py --run-name damage-mb-03 --checkpoint-run phaseb-ian-01      # fine-tuned
modal volume get disaster-lens-results /alignment/damage-mb-03 outputs/drone/runs/
```

Scenario A: drone image with ground-truth alignment; B: drone image aligned by step 2; C (`D` in the files): post-event
satellite image with ground-truth alignment. All scored against the drone-view labels.

## 4. Fine-tuning on Hurricane Ian

```bash
modal run --detach drone/modal_crasar_prep.py --limit 1    # smoke test: one mosaic
modal run --detach drone/modal_crasar_prep.py              # all mosaics, CPU; tiles to /data/raw/crasar-ian
uv run python drone/make_split.py                           # Ian + equal share of xBD -> crasar-ian-mix-v1
modal volume put disaster-lens-data outputs/drone/inputs/splits/crasar-ian-mix-v1 /processed/splits/crasar-ian-mix-v1
modal run --detach modal_app.py::damage_live --run-name phaseb-ian-01 --encoder dinov3-vitl --inputs paired \
    --trainable-blocks 4 --init-model vit-finetune-bf16-l4-20261020-01 --split crasar-ian-mix-v1 --precision bf16 \
    --confirm-run
modal run --detach modal_app.py::end_to_end --run-name test-e2e-phaseb-ian-01 --damage-run phaseb-ian-01 \
    --detector-run segmenter-crops-l4-02 --split test --confirm-run                    # xBD test: no forgetting
```

Starts from the reported model (all weights), five epochs, model selected on Ian validation (same hurricane as
training, so its score is optimistic; Mexico Beach is the test). Ten epochs did not help: validation loss rises after
five.

## 5. Report

```bash
uv run --group drone python report/scripts/drone.py   # or: report/scripts/build.py --drone
```

Writes the drone part of `report/bundle` (explorer, data cards, matches, close-ups, examples, alignment and damage
tables); see `report/README.md`. Preview sheets of any site: `uv run --group drone python drone/previews.py
mexico-beach-1 out.jpg` (also `palu-<n>`, `kilauea-<n>`).

# DisasterLens

**Which buildings did a disaster damage?** Given a satellite image of an area before a disaster and one after it,
DisasterLens finds every building and rates it on the four-level xBD scale: **no damage, minor, major, destroyed**.

<p align="center"><img src="assets/scene-explorer.gif" width="520" alt="A wildfire scene: the after image slides over the before image, the detector outlines 72 buildings, then each outline is coloured by its predicted damage class."></p>

| Scenario | Inputs | Macro-F1 (xBD test) |
| --- | --- | ---: |
| **Known footprints** | before and after images, plus pre-event building outlines | **0.802** |
| **Images only** | before and after images; buildings are detected first | **0.574** |

On the buildings the detector finds, damage macro-F1 is **0.806**: the gap between the two scenarios is finding the
buildings, not judging their damage. Every choice was made on a validation split; the official xBD test split
(739 scenes, 53,850 labeled buildings) was scored once, at the end.

## 1. The task

After a hurricane, flood or wildfire, responders need to know which buildings are damaged and which are safe, across
whole cities, within days. Satellites photograph the area before and after the event; doing the comparison by hand
does not scale. The goal here is one damage class per building, from the two images alone or with a known building
map.

## 2. The data

<p align="center"><img src="assets/data.jpg" width="100%" alt="Before and after pairs for wildfire, flood, hurricane wind, tsunami, earthquake and volcano; class shares and split sizes."></p>

[xBD](https://xview2.org/dataset) pairs pre- and post-event satellite images (1024 × 1024 px) and labels every building
outline with a damage class; its train and test splits, used here, cover ten disasters. Three quarters of the buildings are undamaged, so results use
**macro-F1**: the mean of the four per-class F1 scores, where minor damage counts as much as no damage.

## 3. The model

<p align="center"><img src="assets/architecture.jpg" width="100%" alt="Two stages. A: Mask R-CNN on the pre-event image finds building footprints (or known footprints are used). B: when the post-disaster image arrives, one DINOv3 ViT-L encodes both dates; per-date patch embeddings and scene contexts are pooled per footprint and a small head classifies damage."></p>

Two stages, run at different times:

**A · Building identification (pre-event image).** Mask R-CNN with a ResNet-50-FPN backbone, starting from COCO
weights: ResNet-50 → feature pyramid → region proposals → RoIAlign → box head (building vs. background) and mask head
(28 × 28 mask per building). The ResNet stem and first stage stay frozen; stages 2–4, the pyramid, the proposal network
and both heads are fine-tuned on xBD footprints, with new two-class predictors. Two thirds of all buildings cover less
than 32 × 32 px, so the detector is trained and run on 512 px tiles enlarged 2×; that raised mask AP from 0.343 to
0.405 and small-building recall from 44% to 58% (validation). With an existing building map, this stage is skipped.

**B · Damage assessment (when the post-disaster image arrives).**
- **Encode both dates** with one DINOv3 ViT-L/16, pretrained on satellite imagery (SAT-493M): 16 × 16 px patches →
  24 transformer blocks (1024-d, 16 heads, rotary positions) → a 64 × 64 grid of 1024-d patch embeddings per date.
  Blocks 1–20 stay frozen; blocks 21–24 and the final norm (50M of 303M parameters) are fine-tuned.
- **Keep the context of each date:** the mean of all patch embeddings, for the pre- and the post-event image.
- **Collect features per footprint:** RoIAlign pools a 3 × 3 grid of patch embeddings inside each footprint, on both
  dates.
- **Compare and classify:** shared projections to 128-d, then a small MLP (trained for this task) on
  **[post building, post context, post − pre building, post − pre context]** → no damage / minor / major / destroyed.

Damage models tried, in order (test macro-F1 with known footprints): post-image-only ResNet-18 **0.669** → paired
ResNet-18 **0.724** → frozen ViT **0.783** → fine-tuned ViT **0.802**. Comparing before with after, a
satellite-pretrained encoder, and fine-tuning it each helped.

## 4. Results

<p align="center"><img src="assets/results.png" width="100%" alt="Headline scores; a flow of the 53,850 test buildings into found and missed, and found into exact, one class off, further off; a confusion grid; detector metrics and per-class F1."></p>

- **Classification holds up after detection.** 91% of found buildings get their exact class, about the same as with
  known footprints (91%).
- **Detection is the bottleneck.** The detector finds 68% of labeled buildings at its operating threshold and adds
  15,474 detections that match no labeled building. 82% of the missed buildings are undamaged, often small and in
  dense neighbourhoods.
- **Minor damage is the hardest class** (F1 0.66): 28% of minor-damage buildings are rated undamaged.

<p align="center"><img src="assets/examples.jpg" width="100%" alt="Five test tiles (wildfire, flood, hurricane wind, tsunami, earthquake) with before, after, predicted and reference damage."></p>

## 5. What moves the damage score

<p align="center"><img src="assets/change.jpg" width="100%" alt="Probe 1: with the pre-event image given twice, predicted damage nearly disappears. Probe 2: encoder feature change lines up with burned houses. Panel 3: predicted severity against simple image-change statistics by disaster."></p>

- **It reads change, not context.** Given the pre-event image as both "before" and "after", the model rates almost
  nothing as damaged: destroyed buildings drop from 93.5% to 1.3% predicted damaged, minor from 72.4% to 0.4%. It does
  not guess damage from what a neighbourhood looked like.
- **Where damage is visible from above, predictions follow it.** For wildfires, predicted severity rises with colour
  and edge change inside the footprint and with lost greenness around it, tracking the labels. Flood and earthquake
  damage barely changes roof pixels, which is where the model is weakest.

These are probes and associations, not a complete explanation of the network.

## 6. Next steps: from benchmark to deployment

<p align="center"><img src="assets/next-steps.jpg" width="100%" alt="Deployment workflow: post-disaster image with metadata, search a region of a reference archive, match keypoints with RANSAC, gate on match quality, then assess damage on aligned reference footprints, or fall back to detection. A real SIFT match on the test wildfire pair."></p>

**Registration is the first problem in the field.** With known footprints the model scores 0.802; detecting buildings
from the images alone gives 0.574. In xBD the before and after images arrive co-registered with their footprints (the
measured residual is a few pixels, and correcting it did not improve the score), so the gap there is detection. A real
deployment gets a post-disaster image with a GPS fix and camera metadata, not aligned to anything. The workflow:

1. Use the metadata to bound a search region in an archive of pre-event imagery and building footprints.
2. Match local keypoints between the post-disaster image and candidate reference tiles (classic SIFT/ORB, or learned
   matchers such as SuperPoint + LightGlue), and fit the alignment with RANSAC.
3. If enough matches are consistent, warp the reference footprints onto the new image and run the damage model on
   them: the 0.802 path. If not, fall back to detecting buildings and flag the area for review.

On the wildfire pair above, SIFT finds 123 descriptor matches, 68 consistent after RANSAC: all on intact roads and
houses, almost none in the burned area. The damage itself is what makes registration hard.

Also next:
- **Misalignment tolerance:** shift footprints synthetically to measure how much registration error the damage model
  tolerates.
- **New disasters:** all results come from ten events seen in training; hold out whole events.
- **Detection:** 78% of missed buildings already have a correct outline with a low score; a pilot that re-scores
  detections raised end-to-end validation macro-F1 from 0.566 to 0.583.
- **Subtle damage:** minor vs. no damage is the main confusion; pool features over the building mask, not its box.
- One training run per model; labels describe what is visible from above, not structural safety.

## Repository layout

```
src/disaster_lens/
  data/        xBD extraction, building manifest, scene-grouped split, building index, footprint rasterization, datasets
  damage/      encoders (ResNet-18, DINOv3 ViT-L), feature cache, heads, DamageClassifier, trainers, exports, predictors
  detection/   Mask R-CNN, compact masks, crop training, tiled inference, trainer/exporter, metrics, COCO weights,
               candidate re-scoring (verifier pilot)
  end_to_end/  damage predictions on detected and annotated footprints, scoring, no-change probe
modal_app.py   runs any stage on a Modal L4 GPU
assets/        README figures
tests/         CPU unit and workflow tests on synthetic data
```

Damage classifiers are defined by three choices, and any combination trains with one command:

| Choice | Options |
| --- | --- |
| `--encoder` | `resnet18`, `dinov3-vitl` |
| `--inputs` | `post` (post-event image only), `paired` (pre + post) |
| `--trainable-blocks` | `0` (frozen encoder, train the head) or `N` (also train the last N encoder blocks) |

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group vit                       # timm for DINOv3; add --group cloud for Modal
```

**Data.** Download the xBD train and test archives from the [official portal](https://xview2.org/download) into
`datasets/`. xBD is licensed under CC BY-NC-SA 4.0 and is not included here.

```bash
uv run python -m disaster_lens.data.extract --splits train test
uv run python -m disaster_lens.data.manifest --split train                 # labeled buildings
uv run python -m disaster_lens.data.split --seed 42                        # scene-grouped train/validation split
uv run python -m disaster_lens.data.building_index --data-dir datasets \
    --assignments datasets/processed/splits/seed42/scene_assignments.csv \
    --output-dir datasets/processed/localization/seed42-v1                 # all pre-event footprints per scene
uv run python -m disaster_lens.data.manifest --split test                  # test: labeled buildings
uv run python -m disaster_lens.data.building_index --source-split test     # test: footprints (localization/test-v1)
```

**Pretrained weights.** ResNet-18 comes from torchvision. The others are local files:

```bash
uv run python -m disaster_lens.detection.weights                          # COCO Mask R-CNN, writes provenance.json
uvx hf download timm/vit_large_patch16_dinov3.sat493m --revision ce148bd5ed3de1eb75ebdccafcb2c280bb3e8f7c \
    --local-dir datasets/pretrained/dinov3-vitl16-sat                      # DINOv3 (accept its license on Hugging Face)
```

## Reproduce

Commands below run locally (`--device cuda` on a GPU); every stage also has a Modal workflow (`modal run --detach modal_app.py::<workflow>`; see
the docstring of `modal_app.py`). Runs never overwrite an existing output directory.

**Damage models** (`S=datasets/processed/splits/seed42`, `V=datasets/pretrained/dinov3-vitl16-sat`):

```bash
# Post-only baseline: frozen ResNet-18, linear head on the post-event image
python -m disaster_lens.damage.train_live train --encoder resnet18 --inputs post --trainable-blocks 0 \
    --head-lr 0.001 --data-dir datasets --split-dir $S --output-dir runs/post-only --device cuda

# Paired CNN and frozen ViT: cache frozen features once, then train the fusion head on them
python -m disaster_lens.damage.feature_cache --encoder resnet18 --cache-dir runs/features/resnet18 --batch-size 8 --device cuda
python -m disaster_lens.damage.train_cached --train-cache runs/features/resnet18/train \
    --validation-cache runs/features/resnet18/validation --output-dir runs/paired-cnn --batch-size 8 --device cuda
python -m disaster_lens.damage.feature_cache --encoder dinov3-vitl --encoder-dir $V --cache-dir runs/features/dinov3 --device cuda
python -m disaster_lens.damage.train_cached --train-cache runs/features/dinov3/train \
    --validation-cache runs/features/dinov3/validation --output-dir runs/frozen-vit --batch-size 8 --device cuda

# Fine-tuned ViT: start from the frozen ViT head and also train the last 4 blocks
python -m disaster_lens.damage.train_live train --encoder dinov3-vitl --encoder-dir $V --inputs paired \
    --trainable-blocks 4 --init-head runs/frozen-vit/best.pt --precision bf16 \
    --data-dir datasets --split-dir $S --output-dir runs/fine-tuned-vit --device cuda

# Validation predictions
python -m disaster_lens.damage.export_cached --run-dir runs/paired-cnn --validation-cache runs/features/resnet18/validation
python -m disaster_lens.damage.train_live export --run-dir runs/fine-tuned-vit --data-dir datasets --manifest $S/validation.csv
```

**Detector** (`I=datasets/processed/localization/seed42-v1`; SHA-256 from the weights' `provenance.json`):

```bash
python -m disaster_lens.detection.train train --fresh-training --coco-weights datasets/pretrained/maskrcnn-resnet50-fpn-coco-v1/weights.pt \
    --weights-sha256 $SHA --epochs 5 --index-dir $I --output-dir runs/detector --device cuda
python -m disaster_lens.detection.train train --resume-from runs/detector --resume-checkpoint best --additional-epochs 1 \
    --allow-protocol-change --crop-size 512 --crops-per-scene 4 --validation-subset 100 --validation-tile-size 512 \
    --index-dir $I --output-dir runs/detector-2x --device cuda
python -m disaster_lens.detection.train export --run-dir runs/detector-2x --tile-size 512 --index-dir $I --device cuda
```

**End to end** (both scenarios in one pass, then scoring):

```bash
python -m disaster_lens.end_to_end.run --checkpoint runs/fine-tuned-vit/best.pt \
    --saved-predictions runs/fine-tuned-vit/validation_predictions/predictions.csv \
    --manifest $S/validation.csv --data-dir datasets --export-dir runs/detector-2x/validation_predictions_tile512 \
    --output-dir runs/end-to-end --device cuda
python -m disaster_lens.end_to_end.score --run-dir runs/end-to-end --output runs/end-to-end/scores.json
```

The end-to-end run first checks that the damage model reproduces its own recorded validation results on the
annotated footprints and stops if it does not. For a cached-feature head (paired CNN, frozen ViT) pass the head's
`best.pt` and, for DINOv3, `--encoder-dir`.

**Test split** (once, after all choices are made). The export refuses test scenes that overlap the detector's
training or validation scenes; the test split has no recorded damage predictions, so the reference check is skipped:

```bash
python -m disaster_lens.detection.train export --run-dir runs/detector-2x --split test --tile-size 512 \
    --index-dir datasets/processed/localization/test-v1 --device cuda
python -m disaster_lens.end_to_end.run --checkpoint runs/fine-tuned-vit/best.pt --skip-reference-check \
    --manifest datasets/processed/test_buildings.csv --data-dir datasets \
    --export-dir runs/detector-2x/test_predictions_tile512 --output-dir runs/end-to-end-test --device cuda
```

## Tests

```bash
uv run --group vit --group cloud python -m unittest discover -s tests
```

Tests run on CPU with synthetic images and tiny models: every encoder × inputs × trainable-blocks combination,
detector training/export/tiling, end-to-end runs and scoring, and the Modal runner (without Modal calls).

## License

Code: MIT (see `LICENSE`); `LICENSES/` holds third-party notices (torchvision-derived Mask R-CNN code,
BSD-3-Clause). Data: xBD, CC BY-NC-SA 4.0, not included. The figures in `assets/` contain xBD imagery and labels and
are shared under CC BY-NC-SA 4.0 (see `LICENSES/xBD-imagery.md`). DINOv3 weights are subject to Meta's DINOv3
license.

xBD: Gupta et al., *xBD: A Dataset for Assessing Building Damage from Satellite Imagery*, CVPR Workshops 2019.

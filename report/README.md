# Report pipeline

One pipeline produces everything the README and the website show: numbers, scene imagery, outlines and figures.
Its output, the **bundle** in `report/bundle/`, is committed. The README figures in `assets/` are rendered from the
bundle, and the website copies the bundle (pinned to a commit of this repository) and draws its own interactive
components from it.

```
report/
  scripts/   build steps (Python), run from the repository root
  figures/   README figure templates (HTML/CSS) and render.mjs
  bundle/    generated output, committed
    site.json        all numbers and outlines the website report reads
    manifest.json    every file's SHA-256 and size, the source commit, build time, schema version
    data/            intermediate numbers (metrics, recall, flow, probes, change statistics, scene lists)
    images/          scene crops, overlays, 448 px example tiles, architecture parts, feature maps, card cover
    scenes/          compact building outlines per explorer scene
    media/           scene-explorer.gif for the README
```

## Rebuilding

```bash
uv run --group notebook python report/scripts/build.py   # metrics + scenes → data, images, scenes, site.json, manifest
node report/figures/render.mjs                           # README figures → assets/ (needs local Google Chrome)
```

Optional steps, only when their inputs change:

```bash
uv run --group notebook --group vit python report/scripts/build.py --features   # DINOv3 feature images (heavy)
uv run --group notebook python report/scripts/build.py --registration           # keypoint example (OpenCV via uvx)
```

## Inputs (local, not in Git)

The steps read model outputs and imagery that are not committed (size and licensing):

| Step | Reads |
| --- | --- |
| `metrics.py` | end-to-end runs for the four damage models and the no-change probe (`outputs/modal/test-e2e-*`, `outputs/local/test/*`), detector exports and error analyses (`outputs/modal/segmenter-*`, `outputs/analysis/instance-errors-*`), training results (`outputs/modal/*.result.json`); validation imagery for the change statistics (cached per building in `outputs/analysis/report/`) |
| `scenes.py` | the test-split detector export and fine-tuned ViT predictions; xBD test images and labels (`datasets/raw/test/`); building manifests (`datasets/processed/`) |
| `features.py` | the local DINOv3 weights (`datasets/pretrained/`), PyTorch |
| `registration.py` | xBD test images and labels; OpenCV through `uvx` |

How those outputs are produced is described in the main README (Reproduce). Scene choices follow fixed rules in
`scenes.py` (no hand-picking): the README wildfire scene, then per disaster type the test scenes with the most damaged
buildings, cropped to their densest window.

## Using the bundle elsewhere

- `site.json` has a `schema_version`. Consumers (the website components) should check it and fail loudly on a version
  they do not know.
- Imagery contains xBD data (CC BY-NC-SA 4.0); keep the attribution with any copy.
- After rebuilding, commit the bundle and the README figures together, so a commit always pairs code, numbers and
  images. The manifest's `source_commit` is the commit the bundle was built from.

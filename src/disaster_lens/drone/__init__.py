"""Drone imagery (phase two): align a drone orthomosaic to a satellite pre-event reference, then assess damage.

Modules: `geo` (GeoTIFF reading on a UTM grid), `crasar` (CRASAR-U-DROIDs data: images, footprints, ground-truth
positions), `alignment` (masks, fits, correction fields, warping), `inputs` and `evaluate` (command-line tools).
Optional dependencies: `uv sync --group drone`. Workflow and Modal jobs: drone/README.md.
"""

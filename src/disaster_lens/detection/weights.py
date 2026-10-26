"""Download and verify the COCO-pretrained Mask R-CNN weights the detector starts from.

    python -m disaster_lens.detection.weights --output-dir datasets/pretrained/maskrcnn-resnet50-fpn-coco-v1

Writes `weights.pt` and `provenance.json` (source URL, size and full SHA-256). The eight-character hash prefix in the
official torchvision filename is checked; pass the recorded SHA-256 to `detection.train --weights-sha256`.
"""

import argparse
import hashlib
import json
import tempfile
import time
import urllib.request
from pathlib import Path

NAME = "maskrcnn-resnet50-fpn-coco-v1"
URL = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
HASH_PREFIX = "bf2d0c1e"
WEIGHTS_FILE = "weights.pt"
MAX_BYTES = 256 * 1024**2


def download(output_dir: Path, *, timeout_seconds: int = 600) -> dict:
    """HTTPS download with a size/time limit and no retries; publishes the directory only when verified."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".coco-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "weights"
        staging.mkdir()
        digest, size, started = hashlib.sha256(), 0, time.monotonic()
        with urllib.request.urlopen(URL, timeout=30) as response, (staging / WEIGHTS_FILE).open("xb") as stream:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES or time.monotonic() - started > timeout_seconds:
                    raise TimeoutError("Download exceeded its size or time limit.")
                digest.update(chunk)
                stream.write(chunk)
        info = {"model": NAME, "source_url": URL, "file": WEIGHTS_FILE, "size": size, "sha256": digest.hexdigest()}
        if not info["sha256"].startswith(HASH_PREFIX):
            raise ValueError("Downloaded weights do not match the official filename hash prefix.")
        (staging / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
        staging.rename(output_dir)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/pretrained") / NAME)
    print(json.dumps(download(parser.parse_args().output_dir), indent=2))


if __name__ == "__main__":
    main()

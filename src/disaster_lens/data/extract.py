"""Extract official xBD train/test archives without overwriting data.

Run from the repository root:
    uv run python -m disaster_lens.data.extract --splits train test
"""

import argparse
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

def extract_split(
    archive_path: Path,
    output_dir: Path,
    split: str,
) -> Path:
    """Extract into staging, publishing the split only after successful completion.

    Accept only regular files/directories under the expected split root.
    Existing destinations are never overwritten. A failed extraction is cleaned up.
    """
    destination = output_dir / split
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{split}-", dir=output_dir) as staging:
        staging_path = Path(staging)
        file_count = 0
        print(f"Extracting {split} ...", flush=True)
        with tarfile.open(archive_path, mode="r|*") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or not name.parts
                    or name.parts[0] != split
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError(f"Unexpected archive member: {member.name}")
                archive.extract(member, path=staging_path, filter="data")
                file_count += member.isfile()

        extracted = staging_path / split
        for directory in ("images", "labels", "targets"):
            if not (extracted / directory).is_dir():
                raise ValueError(f"Missing expected directory: {split}/{directory}")
        extracted.rename(destination)

    print(f"Extracted {file_count:,} files to {destination}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "test"), default=["train", "test"]
    )
    args = parser.parse_args()
    for split in args.splits:
        extract_split(
            archive_path=args.data_dir / f"{split}_images_labels_targets.tar",
            output_dir=args.data_dir / "raw",
            split=split,
        )


if __name__ == "__main__":
    main()

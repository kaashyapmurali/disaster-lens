"""Extraction tests use tiny synthetic archives, not the real dataset."""

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from disaster_lens.data.extract import extract_split


class ExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "sample.tar.gz"
        self.output = self.root / "raw"

    def make_archive(self, extra: tarfile.TarInfo | None = None) -> None:
        with tarfile.open(self.archive, "w:gz") as archive:
            for directory in ("images", "labels", "targets"):
                member = tarfile.TarInfo(f"train/{directory}")
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            content = b'{"features": {}}'
            member = tarfile.TarInfo("train/labels/example.json")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
            if extra is not None:
                archive.addfile(extra)

    def test_extract_and_refuse_overwrite(self) -> None:
        self.make_archive()
        destination = extract_split(self.archive, self.output, "train")
        self.assertEqual(
            (destination / "labels/example.json").read_bytes(),
            b'{"features": {}}',
        )
        with self.assertRaises(FileExistsError):
            extract_split(self.archive, self.output, "train")

    def test_invalid_members_are_rejected_and_staging_is_cleaned(self) -> None:
        link = tarfile.TarInfo("train/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/tmp"
        for member in (
            tarfile.TarInfo("train/../../escape"),
            tarfile.TarInfo("/absolute"),
            tarfile.TarInfo("test/wrong-split"),
            link,
        ):
            with self.subTest(name=member.name):
                self.make_archive(member)
                with self.assertRaisesRegex(ValueError, "Unexpected archive member"):
                    extract_split(self.archive, self.output, "train")
                self.assertEqual(list(self.output.iterdir()), [])

    def test_incomplete_archive_is_not_published(self) -> None:
        with tarfile.open(self.archive, "w:gz") as archive:
            member = tarfile.TarInfo("train/images")
            member.type = tarfile.DIRTYPE
            archive.addfile(member)
        with self.assertRaisesRegex(ValueError, "Missing expected directory"):
            extract_split(self.archive, self.output, "train")
        self.assertEqual(list(self.output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

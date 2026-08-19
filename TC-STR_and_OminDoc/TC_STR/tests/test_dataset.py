import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from tc_str_bench.dataset import build_manifest


def png(width=2, height=3):
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
    raw = b"\x00" + b"\x00\x00\x00" * width
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw * height)) + chunk(b"IEND", b"")


class DatasetTests(unittest.TestCase):
    def test_first_tab_and_order_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            for name in ("a.png", "b.png", "c.png"):
                (root / "images" / name).write_bytes(png())
            (root / "test_labels.txt").write_text(
                "images/a.png\t甲\t乙\nimages/b.png\tABC 123\nimages/c.png\t長文字，測試！\n",
                encoding="utf-8",
            )
            one = build_manifest(root, expected=3)
            two = build_manifest(root, expected=3)
            self.assertEqual(one["samples"][0]["ground_truth"], "甲\t乙")
            self.assertEqual([s["index"] for s in one["samples"]], [0, 1, 2])
            self.assertEqual(one["dataset_fingerprint"], two["dataset_fingerprint"])
            self.assertEqual((one["samples"][0]["width"], one["samples"][0]["height"]), (2, 3))

    def test_missing_tab_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_labels.txt").write_text("bad\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_manifest(root, expected=1)


if __name__ == "__main__":
    unittest.main()

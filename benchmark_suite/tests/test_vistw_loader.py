import io

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from ocf_benchmark.datasets.vistw_mcq import SUBJECTS, load_samples


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(output, format="PNG")
    return output.getvalue()


def test_local_parquet_loader_has_globally_unique_sample_ids(tmp_path):
    snapshot = tmp_path / "snapshot"
    row = {
        "qid": "0",
        "image": {"bytes": _png_bytes(), "path": None},
        "source": "test",
        "question": "題目",
        "A": "甲",
        "B": "乙",
        "C": "丙",
        "D": "丁",
        "answer": "A",
    }
    for subject in SUBJECTS:
        destination = snapshot / subject / "test-00000-of-00001.parquet"
        destination.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([row]), destination)

    samples = list(
        load_samples(
            "miulab/vistw-mcq",
            "fixed-revision",
            tmp_path / "images",
            split="test",
            snapshot=snapshot,
        )
    )
    assert len(samples) == 21
    assert len({sample.sample_id for sample in samples}) == 21
    assert samples[0].sample_id == "accounting:0"
    assert samples[-1].sample_id == "veterinary_medicine:0"
    assert all(sample.image_path.exists() for sample in samples)

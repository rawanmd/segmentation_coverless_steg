# Segmentation Coverless Steganography (SAM)

Modular coverless image steganography built on Segment Anything Model (SAM) masks stored in HDF5 (`.h5`). Precomputing SAM masks once on GPU lets you write and benchmark new `CoverlessStegPort` adapters in seconds on CPU.

## Repository Layout

- `coverless_steg.py` — Flat module containing `CoverlessStegPort`, `SAMAdapter`, `BBoxDistanceAdapter`, in-memory noise/JPEG attack utilities, `generate_sam_h5`, and `test_adapter_variance`.
- `01_generate_sam_h5.ipynb` — Colab GPU setup to generate the 3 HDF5 mask datasets (`clean`, `noise`, `compression`).
- `02_adapters_and_variance_test.ipynb` — CPU-fast workflow: embed/extract with `SAMAdapter`, write a new adapter in ~10 lines, and measure bit-error variance across attacks.
- `03_visualization.ipynb` — Visualize in-memory attacks, SAM mask overlays, and `SAMAdapter` grid features without temporary disk files.
- `tests/test_coverless_steg.py` — Unit and H5 integration tests.

## Precomputed HDF5 Datasets (`dataset/`)

Each file stores 500 COCO `train2017` groups (`"0000"`..`"0499"`):

| File | Description |
| --- | --- |
| `dataset/SAM_outputs_clean.h5` | SAM `vit_b` masks on unmodified RGB images |
| `dataset/SAM_outputs_noise.h5` | SAM masks after in-memory Gaussian noise (`mean=0.0`, `var=0.001`) |
| `dataset/SAM_outputs_compression.h5` | SAM masks after in-memory JPEG compression (`quality=90`) |

Each group carries `attrs` (`image_path`, `mask_count`, `height`, `width`, `r_mean`) and gzip-compressed datasets (`segmentation`, `bbox`, `area`, `predicted_iou`, `point_coords`, `stability_score`, `crop_box` when `mask_count > 0`).

## Writing a New `CoverlessStegPort` Adapter in ~10 Lines

Subclass `CoverlessStegPort` and implement `encode_entry(entry) -> str`:

```python
import numpy as np
from coverless_steg import CoverlessStegPort, binary_to_string, string_to_binary

class MaskAreaAdapter(CoverlessStegPort):
    def encode_entry(self, entry) -> str:
        areas = entry.get("area")
        if areas is None or len(areas) < 2:
            return ""
        return "".join("1" if d > 0 else "0" for d in np.diff(areas))

adapter = MaskAreaAdapter("dataset/SAM_outputs_clean.h5")
keys, win, pad = adapter.embed(string_to_binary("Hello"))
assert binary_to_string(adapter.extract(keys, win, pad)) == "Hello"
```

## Running the Variance Tester & Unit Tests

Compare input vs. extracted bits across clean and attacked HDF5 mask sets:

```python
from coverless_steg import SAMAdapter, test_adapter_variance

adapter = SAMAdapter("dataset/SAM_outputs_clean.h5", grid_size=16, tolerance=0.02)
report = test_adapter_variance(
    adapter,
    clean_source="dataset/SAM_outputs_clean.h5",
    extraction_sources={
        "clean": "dataset/SAM_outputs_clean.h5",
        "noise": "dataset/SAM_outputs_noise.h5",
        "compression": "dataset/SAM_outputs_compression.h5",
    },
    messages=["HelloWorld", "Coverless", "SAM Steg!"],
)
print(report)
```

Run tests locally:

```bash
pytest -q
```

# Segmentation Coverless Steganography (SAM)

Modular coverless image steganography built around a pure hexagonal `CoverlessStegPort` (`embed` + `extract`) and an optional `IndexedStegAdapter` over Segment Anything Model (SAM) masks stored in HDF5 (`.h5`).

## Repository Layout

- `coverless_steg.py` — Flat module containing `CoverlessStegPort`, `IndexedStegAdapter`, `SAMAdapter`, `BBoxDistanceAdapter`, in-memory noise/JPEG attack utilities, `generate_sam_h5`, and `test_adapter_variance`.
- `01_generate_sam_h5.ipynb` — Colab GPU setup to generate the 3 HDF5 mask datasets (`clean`, `noise`, `compression`).
- `02_adapters_and_variance_test.ipynb` — CPU-fast workflow: pure `CoverlessStegPort` adapters, `IndexedStegAdapter` subclasses in ~8 lines, and bit-error variance benchmarking across attacks.
- `03_visualization.ipynb` — Visualize in-memory attacks, SAM mask overlays, and `SAMAdapter` grid features without temporary disk files.
- `tests/test_coverless_steg.py` — Unit and HDF5 integration tests.
- `pyproject.toml` & `uv.lock` — Reproducible dependencies for one-command test execution via `uv`.

## Precomputed HDF5 Datasets (`dataset/`)

Tracked via Git LFS (`.gitattributes` tracks `*.h5` and `*.pth`), each file stores 500 COCO `train2017` groups (`"0000"`..`"0499"`):

| File | Description |
| --- | --- |
| `dataset/SAM_outputs_clean.h5` | SAM `vit_b` masks on unmodified RGB images |
| `dataset/SAM_outputs_noise.h5` | SAM masks after in-memory Gaussian noise (`mean=0.0`, `var=0.001`) |
| `dataset/SAM_outputs_compression.h5` | SAM masks after in-memory JPEG compression (`quality=90`) |

Each group carries `attrs` (`image_path`, `mask_count`, `height`, `width`, `r_mean`) and gzip-compressed datasets (`segmentation`, `bbox`, `area`, `predicted_iou`, `point_coords`, `stability_score`, `crop_box` when `mask_count > 0`).

> **Note:** `.h5` files only cache dataset image masks — **embedding new messages never requires regenerating `.h5` files**. You only regenerate `.h5` files if you swap the image dataset, SAM checkpoint, or attack parameters.

## Writing a New Adapter (Two Ways)

### 1. Any Custom Method (No Masks or `.h5` Needed) — `CoverlessStegPort`

`CoverlessStegPort` only defines `embed` and `extract`, with zero mask, HDF5, or index assumptions:

```python
from coverless_steg import CoverlessStegPort, binary_to_string, string_to_binary

class CarrierLookupAdapter(CoverlessStegPort):
    def __init__(self, carriers: dict[str, str]):
        self.carriers = carriers  # {carrier_id: bitstring}

    def embed(self, binary_msg: str) -> list[str] | None:
        inv = {bits: cid for cid, bits in self.carriers.items()}
        keys = [inv.get(binary_msg[i : i + 4]) for i in range(0, len(binary_msg), 4)]
        return None if any(k is None for k in keys) else keys  # type: ignore[return-value]

    def extract(self, payload: list[str], source=None) -> str:
        table = self.carriers if source is None else source
        return "".join(table[cid] for cid in payload)
```

### 2. Index / Feature-Based Method Over SAM `.h5` Masks (~8 Lines) — `IndexedStegAdapter`

For methods that map dataset entries (`.h5`, dicts, or raw images) to bitstrings with adaptive-window prefix matching, subclass `IndexedStegAdapter` and implement `encode_entry(entry) -> str`:

```python
import numpy as np
from coverless_steg import IndexedStegAdapter, binary_to_string, string_to_binary

class MaskAreaAdapter(IndexedStegAdapter):
    def encode_entry(self, entry) -> str:
        areas = entry.get("area")
        if areas is None or len(areas) < 2:
            return ""
        return "".join("1" if d > 0 else "0" for d in np.diff(areas))

adapter = MaskAreaAdapter("dataset/SAM_outputs_clean.h5")
payload = adapter.embed(string_to_binary("Hello"))
assert binary_to_string(adapter.extract(payload)) == "Hello"
```

## Running the Variance Tester & Unit Tests

`test_adapter_variance` benchmarks **any** `CoverlessStegPort` (both pure `embed`/`extract` adapters and `IndexedStegAdapter` subclasses):

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

Run tests locally with `uv`:

```bash
uv run pytest -q
# or explicitly sync the virtual environment first:
uv sync && .venv/bin/pytest -q
```

"""Unit and HDF5 integration tests for coverless_steg and split notebooks."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import warnings

import cv2
import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coverless_steg import (
    BBoxDistanceAdapter,
    CoverlessStegPort,
    H5Entry,
    SAMAdapter,
    apply_gaussian_noise,
    apply_jpeg_comp,
    binary_to_string,
    build_prefix_maps,
    calculate_grid_means,
    compare_neighbors,
    compute_bit_errors,
    compute_mask_matrices,
    generate_sam_h5,
    string_to_binary,
    test_adapter_variance,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_H5 = (
    REPO_ROOT / "dataset" / "SAM_outputs_clean.h5"
    if (REPO_ROOT / "dataset" / "SAM_outputs_clean.h5").exists()
    else REPO_ROOT / "dataset" / "SAM_outputs.h5"
)


# ---------------------------------------------------------------------------
# 1. Binary / text & in-memory attack utilities
# ---------------------------------------------------------------------------


def test_binary_string_roundtrip_and_validation():
    for text in ("", "A", "HelloWorld", "SAM Steg 123!"):
        bits = string_to_binary(text)
        assert len(bits) == 8 * len(text)
        assert binary_to_string(bits) == text

    assert binary_to_string("01001000 01101001") == "Hi"

    with pytest.raises(ValueError):
        string_to_binary("emoji 🚀")
    with pytest.raises(ValueError):
        binary_to_string("0102")


def test_in_memory_gaussian_noise_and_jpeg_no_disk_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = np.zeros((32, 48, 3), dtype=np.uint8)
    img[:, :, 0] = 240  # pure red in RGB
    img[:, :, 1] = 20
    img[:, :, 2] = 10

    # Zero variance returns identical values; seeded noise is deterministic.
    assert np.array_equal(apply_gaussian_noise(img, variance=0.0, rng=7), img)
    n1 = apply_gaussian_noise(img, mean=0.0, variance=0.01, rng=7)
    n2 = apply_gaussian_noise(img, mean=0.0, variance=0.01, rng=7)
    assert n1.shape == img.shape and n1.dtype == np.uint8
    assert np.array_equal(n1, n2)
    assert not np.array_equal(n1, img)

    # JPEG compression preserves RGB channel order (red stays dominant, blue stays low).
    comp = apply_jpeg_comp(img, quality=90)
    assert comp.shape == img.shape and comp.dtype == np.uint8
    assert float(np.mean(comp[:, :, 0])) > 220.0
    assert float(np.mean(comp[:, :, 2])) < 30.0

    # Ensure no temp files were written anywhere in cwd or /test.jpg.
    assert list(tmp_path.iterdir()) == []
    assert not Path("/test.jpg").exists()

    with pytest.raises(ValueError):
        apply_gaussian_noise(img, variance=-0.1)
    with pytest.raises(ValueError):
        apply_jpeg_comp(img, quality=105)


# ---------------------------------------------------------------------------
# 2. CoverlessStegPort contract, trailing zeros, and short-chunk handling
# ---------------------------------------------------------------------------


class DictBitsAdapter(CoverlessStegPort):
    """Minimal 4-line custom adapter for deterministic port testing."""

    def encode_entry(self, entry) -> str:
        return str(entry.get("bits", ""))


def test_embed_extract_roundtrip_and_trailing_zeros():
    clean = {
        "k0": {"bits": "101000"},
        "k1": {"bits": "110100"},
        "k2": {"bits": "001100"},
        "k3": {"bits": "011000"},
    }
    adapter = DictBitsAdapter(clean)

    # Case 1: num_zeros == 0 (exact multiple of window_size=4 -> 8 bits across 2 keys)
    msg_exact = "10101101"
    keys, win, pad = adapter.embed(msg_exact, max_window_size=4)
    assert keys == ["k0", "k1"]
    assert win == 4 and pad == 0
    assert adapter.extract(keys, win, pad) == msg_exact

    # Case 2: num_zeros > 0 (6-bit message with window_size=4 -> last chunk '11' padded to '1100', pad=2)
    msg_padded = "101000"
    # Force window_size=4: chunk0='1010' (k0), chunk1='00' -> '0000' (not in index) -> steps down to w=3:
    # At w=3: chunk0='101' (k0), chunk1='000' (not in index) -> steps down to w=2:
    # Let's test explicit padding where chunk1 + '00' matches k2 ('0011' vs '011000'):
    msg_pad_match = "110101"  # w=4: chunk0='1101' (k1), chunk1='01'->'0100' (no); w=4 for '11010' (5 bits):
    # chunk0='1101' (k1), chunk1='0' -> '0000' (no). Let's add '010000' so w=4 pads 2 zeros!
    clean["k4"] = {"bits": "010000"}
    adapter = DictBitsAdapter(clean)
    msg_five = "110101"  # len=6 -> w=4: chunk0='1101' (k1), chunk1='01' -> '0100' (k4), pad=2!
    keys2, win2, pad2 = adapter.embed(msg_five, max_window_size=4)
    assert keys2 == ["k1", "k4"]
    assert win2 == 4 and pad2 == 2
    assert adapter.extract(keys2, win2, pad2) == msg_five

    # Empty message and unmatchable message edge cases
    assert adapter.embed("") == ([], 0, 0)
    assert adapter.extract([], 0, 0) == ""

    no_match_adapter = DictBitsAdapter({"only_zeros": {"bits": "0000"}})
    failed = no_match_adapter.embed("1111")
    assert failed == ([], -1, -1)
    assert no_match_adapter.extract(*failed) == ""

    with pytest.raises(ValueError):
        adapter.embed("10a1")


def test_short_extracted_chunk_warns_and_preserves_alignment():
    clean = {"a": {"bits": "1111"}, "b": {"bits": "1100"}}
    # Simulate an attack that drops masks on 'a' (1 bit instead of 4) and 'b' (2 bits instead of 4).
    attacked = {"a": {"bits": "1"}, "b": {"bits": "11"}}
    adapter = DictBitsAdapter(clean)

    msg = "111111"  # w=4, pad=2: 'a' ('1111') + 'b' ('11' padded to '1100')
    keys, win, pad = adapter.embed(msg, max_window_size=4)
    assert (keys, win, pad) == (["a", "b"], 4, 2)

    with pytest.warns(RuntimeWarning, match="right-padding"):
        ext = adapter.extract(keys, win, pad, source=attacked)

    # 'a' padded -> '1000', 'b' padded -> '1100', strip pad=2 -> '100011' (chunk 'b' stays aligned!).
    assert ext == "100011"
    assert len(ext) == len(msg)


def test_legacy_json_index_compatibility():
    legacy_index = {"1011": [{"image path": "img_a.jpg", "pix": 100}]}
    pmaps = build_prefix_maps(legacy_index)
    adapter = DictBitsAdapter({"img_a.jpg": {"bits": "1011"}})
    keys, win, pad = adapter.embed("101", prefix_maps=pmaps)
    assert keys == ["img_a.jpg"] and win == 3 and pad == 0


# ---------------------------------------------------------------------------
# 3. Grid helpers, SAMAdapter, BBoxDistanceAdapter, and generate_sam_h5
# ---------------------------------------------------------------------------


def test_grid_helpers_match_reference_loop_and_zero_masks():
    rng = np.random.default_rng(123)
    seg = rng.random((7, 37, 53)) > 0.7  # non-divisible H=37, W=53
    count_mat, bin_mat = compute_mask_matrices(seg, 37, 53)

    grid_size = 8
    ref = np.zeros((grid_size, grid_size), dtype=np.float64)
    for r in range(grid_size):
        for c in range(grid_size):
            y1, y2 = r * 37 // grid_size, (r + 1) * 37 // grid_size
            x1, x2 = c * 53 // grid_size, (c + 1) * 53 // grid_size
            ref[r, c] = np.mean(count_mat[y1:y2, x1:x2])

    vec = calculate_grid_means(count_mat, grid_size=grid_size)
    assert np.allclose(vec, ref, atol=1e-12)
    bits = compare_neighbors(vec, tolerance=0.02)
    assert len(bits) == 2 * grid_size * (grid_size - 1)

    # Zero-mask entry handling in SAMAdapter & BBoxDistanceAdapter
    sam_count = SAMAdapter({"empty": {"mask_count": 0, "height": 32, "width": 32}}, grid_size=4, mode="count")
    sam_bin = SAMAdapter({"empty": {"mask_count": 0, "height": 32, "width": 32}}, grid_size=4, mode="binary")
    assert sam_count.encode_entry({"mask_count": 0}) == "0" * 24
    assert sam_bin.encode_entry({"mask_count": 0}) == "0" * 24
    assert BBoxDistanceAdapter().encode_entry({"bbox": [[10, 20, 5, 5]]}) == ""


def test_generate_sam_h5_and_raw_image_fallback(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(3):
        arr = np.full((32, 48, 3), 50 * (i + 1), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{i:04d}.jpg"), arr)

    class DummyMaskGen:
        def generate(self, image_rgb: np.ndarray):
            h, w = image_rgb.shape[:2]
            m1 = np.zeros((h, w), dtype=bool)
            m1[: h // 2, : w // 2] = True
            m2 = np.zeros((h, w), dtype=bool)
            m2[h // 2 :, w // 2 :] = True
            val = int(image_rgb[0, 0, 0])
            return [
                {"segmentation": m1, "bbox": np.array([0, 0, 10, 10]), "area": np.int64(100 + val)},
                {"segmentation": m2, "bbox": np.array([val * 3, 0, 10, 10]), "area": np.int64(200)},
            ]

    mg = DummyMaskGen()
    h5_dir = tmp_path / "out_h5"
    h5_dir.mkdir()
    # Passing directory auto-names SAM_outputs_noise.h5 with zero whitespace
    out_path = generate_sam_h5(h5_dir, img_dir, mg, with_noise=True, num_images=3)
    assert out_path.name == "SAM_outputs_noise.h5"
    assert " " not in out_path.name

    with h5py.File(out_path, "r") as f:
        assert sorted(f.keys()) == ["0000", "0001", "0002"]
        e0 = H5Entry(f["0000"], "0000")
        assert e0["height"] == 32 and e0["width"] == 48 and e0["mask_count"] == 2
        assert "r_mean" in e0 and e0["segmentation"].shape == (2, 32, 48)

    # Positional mask_generator fallback on raw images
    raw_adapter = SAMAdapter(mg, grid_size=4)
    idx = raw_adapter.build_index(img_dir)
    assert len(idx) >= 1


# ---------------------------------------------------------------------------
# 4. Variance Tester math (clean=0, simulated perturbations, length mismatch)
# ---------------------------------------------------------------------------


def test_compute_bit_errors_and_variance_report_math():
    assert compute_bit_errors("", "") == (0, 0.0)
    assert compute_bit_errors("", "10") == (2, 1.0)
    assert compute_bit_errors("1010", "1010") == (0, 0.0)
    # Length mismatch: 1 mismatch in prefix + 2 missing bits = 3 errors -> 3/4 = 0.75
    assert compute_bit_errors("1010", "11") == (3, 0.75)
    # Extracted longer than expected: 0 prefix mismatch + 2 extra bits = 2 errors -> 2/4 = 0.5
    assert compute_bit_errors("1010", "101011") == (2, 0.5)

    clean_src = {"k0": {"bits": "1010"}, "k1": {"bits": "0101"}}
    noisy_src = {"k0": {"bits": "1011"}, "k1": {"bits": "0101"}}  # k0 flips 1 bit, k1 flips 0 bits
    adapter = DictBitsAdapter(clean_src)

    # Message 1 ('1010' -> k0): 1/4 = 0.25 BER on noisy
    # Message 2 ('0101' -> k1): 0/4 = 0.00 BER on noisy
    # Message 3 ('1111' -> unmatchable): embed fails -> BER 1.0 on both
    report = test_adapter_variance(
        adapter,
        clean_src,
        {"clean": clean_src, "noise": noisy_src},
        messages=["1010", "0101"],
        is_binary=True,
    )

    clean_sum = report.conditions["clean"]
    assert clean_sum.mean_ber == 0.0
    assert clean_sum.ber_variance == 0.0
    assert clean_sum.exact_match_rate == 1.0
    assert clean_sum.embed_failure_rate == 0.0

    noise_sum = report.conditions["noise"]
    expected_bers = np.array([0.25, 0.0])
    assert noise_sum.mean_ber == pytest.approx(float(np.mean(expected_bers)))
    assert noise_sum.ber_variance == pytest.approx(float(np.var(expected_bers)))
    assert noise_sum.ber_std == pytest.approx(float(np.std(expected_bers)))
    assert noise_sum.mean_hamming_distance == pytest.approx(0.5)
    assert noise_sum.hamming_variance == pytest.approx(float(np.var([1.0, 0.0])))
    assert noise_sum.exact_match_rate == pytest.approx(0.5)
    assert "noise" in report.format_table()
    assert report.to_dict()["adapter_name"] == "DictBitsAdapter"


# ---------------------------------------------------------------------------
# 5. Real HDF5 Dataset Integration & Split Notebooks Verification
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_H5.exists(), reason="Precomputed SAM HDF5 dataset not present")
def test_real_h5_adapters_and_variance():
    sources = {"clean": REAL_H5}
    for name in ("noise", "compression"):
        candidate = REPO_ROOT / "dataset" / f"SAM_outputs_{name}.h5"
        if candidate.exists():
            sources[name] = candidate

    # 1. Fast full-500-image test with BBoxDistanceAdapter
    bbox_adapter = BBoxDistanceAdapter(REAL_H5)
    bbox_report = test_adapter_variance(
        bbox_adapter, REAL_H5, sources, messages=["Hi", "SAM", "1011011"]
    )
    assert bbox_report.index_keys > 0
    assert bbox_report.conditions["clean"].mean_ber == 0.0
    assert bbox_report.conditions["clean"].ber_variance == 0.0
    assert bbox_report.conditions["clean"].exact_match_rate == 1.0

    # 2. SAMAdapter round-trip + lookup by canonical image_path on real H5 subset & full file
    with h5py.File(REAL_H5, "r") as h5:
        subset = {k: H5Entry(h5[k], k) for k in sorted(h5.keys())[:40]}
        first_path = str(H5Entry(h5["0000"], "0000")["image_path"])
        sam_adapter = SAMAdapter(subset, grid_size=16, tolerance=0.02, mode="count")
        sam_report = test_adapter_variance(
            sam_adapter, subset, {"clean": subset}, messages=["Hi", "10110"]
        )
        assert sam_report.conditions["clean"].exact_match_rate == 1.0
        assert sam_report.conditions["clean"].ber_variance == 0.0

    # Verify H5 lookup also works via canonical image_path
    h5_sam = SAMAdapter(REAL_H5, grid_size=16)
    seq_by_key = h5_sam.extract(["0000"], window_size=12, num_zeros=0)
    seq_by_path = h5_sam.extract([first_path], window_size=12, num_zeros=0)
    assert seq_by_key == seq_by_path and len(seq_by_key) == 12


def test_split_notebooks_are_valid_and_legacy_notebook_removed():
    assert not (REPO_ROOT / "segmentation_coverless_steg_sam.ipynb").exists()

    expected_notebooks = [
        "01_generate_sam_h5.ipynb",
        "02_adapters_and_variance_test.ipynb",
        "03_visualization.ipynb",
    ]
    for nb_name in expected_notebooks:
        nb_path = REPO_ROOT / nb_name
        assert nb_path.is_file(), f"Missing notebook: {nb_name}"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 3
        for idx, cell in enumerate(code_cells):
            src = "".join(cell["source"])
            ast.parse(src, filename=f"{nb_name}#cell-{idx}")
            assert "/test.jpg" not in src

"""Modular coverless image steganography over precomputed SAM HDF5 masks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import glob
import math
import os
from pathlib import Path
import time
from typing import Any
import warnings

import cv2
import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Binary / text & image attack utilities
# ---------------------------------------------------------------------------


def string_to_binary(text: str) -> str:
    """Convert an 8-bit/ASCII text string into a contiguous binary bitstring."""
    if any(ord(ch) > 255 for ch in text):
        raise ValueError("string_to_binary expects 8-bit characters (ord <= 255).")
    return "".join(format(ord(ch), "08b") for ch in text)


def binary_to_string(binary_text: str) -> str:
    """Convert a binary bitstring back into text (8 bits per character)."""
    cleaned = binary_text.replace(" ", "")
    if any(ch not in ("0", "1") for ch in cleaned):
        raise ValueError("binary_to_string expects only '0', '1', or spaces.")
    return "".join(
        chr(int(cleaned[i : i + 8], 2))
        for i in range(0, len(cleaned), 8)
        if cleaned[i : i + 8]
    )


def get_image_paths(directory: str | Path) -> list[str]:
    """Return sorted .jpg image paths inside *directory*."""
    return sorted(glob.glob(os.path.join(str(directory), "*.jpg")))


def apply_gaussian_noise(
    image_rgb: np.ndarray,
    mean: float = 0.0,
    variance: float = 0.001,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Apply additive Gaussian noise in memory to an RGB uint8 image."""
    if variance < 0:
        raise ValueError(f"variance must be >= 0, got {variance}")
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    norm = image_rgb.astype(np.float32) / 255.0
    noisy = np.clip(norm + gen.normal(mean, math.sqrt(variance), norm.shape), 0.0, 1.0)
    return (noisy * 255.0).round().astype(np.uint8)


def apply_jpeg_comp(image_rgb: np.ndarray, quality: int = 90) -> np.ndarray:
    """Apply JPEG compression/decompression in memory to an RGB uint8 image."""
    q = int(quality)
    if not (0 <= q <= 100):
        raise ValueError(f"JPEG quality must be in 0..100, got {quality}")
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg', ...) failed")
    decoded_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise RuntimeError("cv2.imdecode(...) failed")
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Grid & mask encoding helpers (shared by SAMAdapter and visualization)
# ---------------------------------------------------------------------------


def compute_mask_matrices(
    segmentations: np.ndarray | Sequence[Mapping[str, Any]] | None,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (count_matrix, binary_matrix) from (N, H, W) bool array or SAM mask dicts."""
    if segmentations is None or len(segmentations) == 0:
        if height <= 0 or width <= 0:
            raise ValueError(f"Positive height/width required when masks are empty, got {(height, width)}")
        count_matrix = np.zeros((height, width), dtype=np.uint16)
    elif isinstance(segmentations, np.ndarray):
        count_matrix = np.sum(segmentations, axis=0, dtype=np.uint16)
    else:
        first = segmentations[0]
        if isinstance(first, Mapping):
            stacked = np.stack([m["segmentation"] for m in segmentations])
        else:
            stacked = np.asarray(segmentations)
        count_matrix = np.sum(stacked, axis=0, dtype=np.uint16)

    binary_matrix = (count_matrix > 0).astype(np.uint8)
    return count_matrix, binary_matrix


def calculate_grid_means(matrix: np.ndarray, grid_size: int = 16) -> np.ndarray:
    """Compute (grid_size, grid_size) cell means matching row * H // grid_size splits."""
    if grid_size < 1:
        raise ValueError(f"grid_size must be >= 1, got {grid_size}")
    height, width = matrix.shape[:2]
    if height < grid_size or width < grid_size:
        raise ValueError(
            f"matrix shape {(height, width)} must be at least {(grid_size, grid_size)}"
        )
    ys = np.arange(grid_size + 1) * height // grid_size
    xs = np.arange(grid_size + 1) * width // grid_size
    row_sums = np.add.reduceat(matrix.astype(np.float64), ys[:-1], axis=0)
    cell_sums = np.add.reduceat(row_sums, xs[:-1], axis=1)
    return cell_sums / np.outer(np.diff(ys), np.diff(xs))


def compare_neighbors(features: np.ndarray, tolerance: float = 0.02) -> str:
    """Encode horizontal then vertical neighbor differences (> tolerance -> '1', else '0')."""
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 2:
        raise ValueError(f"features must be 2D with shape >= (2, 2), got {features.shape}")
    horiz = (features[:, :-1] - features[:, 1:]) > tolerance
    vert = (features[:-1, :] - features[1:, :]) > tolerance
    bits = np.concatenate([horiz.ravel(), vert.ravel()])
    return "".join("1" if b else "0" for b in bits)


# ---------------------------------------------------------------------------
# Entry & source normalization (H5, dict, or raw image + mask_generator)
# ---------------------------------------------------------------------------


def _infer_entry_meta(data: Mapping[str, Any], field: str) -> int:
    seg = data.get("segmentation")
    if seg is not None:
        shape = seg.shape if hasattr(seg, "shape") else np.asarray(seg).shape
        if field == "mask_count":
            return int(shape[0])
        if shape[0] > 0:
            return int(shape[1] if field == "height" else shape[2])
    if field in ("height", "width") and "crop_box" in data:
        crop = data["crop_box"]
        if crop is not None and len(crop) > 0:
            return int(crop[0][3 if field == "height" else 2])
    return 0


class H5Entry(Mapping[str, Any]):
    """Lazy dict-like view over an HDF5 group with legacy height/width fallbacks."""

    def __init__(self, group: h5py.Group, key: str = ""):
        self._g = group
        self.key = key

    def __getitem__(self, k: str) -> Any:
        attrs, g = self._g.attrs, self._g
        if k in attrs:
            val = attrs[k]
            if isinstance(val, bytes):
                return val.decode("utf-8")
            return val.item() if isinstance(val, np.generic) else val
        if k in g:
            return g[k][:]
        if k in ("height", "width", "mask_count"):
            return _infer_entry_meta(g, k)
        if k == "key":
            return self.key
        raise KeyError(k)

    def __iter__(self) -> Iterator[str]:
        keys = set(self._g.keys()) | set(self._g.attrs.keys()) | {"height", "width", "mask_count", "key"}
        return iter(keys)

    def __len__(self) -> int:
        return len(set(self._g.keys()) | set(self._g.attrs.keys()) | {"height", "width", "mask_count", "key"})


def _normalize_dict_entry(raw: Mapping[str, Any], key: str = "") -> Mapping[str, Any]:
    if isinstance(raw, H5Entry):
        return raw
    entry = dict(raw)
    entry.setdefault("key", key)
    for field in ("height", "width", "mask_count"):
        if field not in entry:
            entry[field] = _infer_entry_meta(entry, field)
    return entry


def _entry_from_image(image_path: str | Path, mask_generator: Any) -> dict[str, Any]:
    if mask_generator is None:
        raise ValueError(f"mask_generator is required to extract masks from raw image {image_path!r}")
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(img_rgb)
    entry: dict[str, Any] = {
        "key": str(image_path),
        "image_path": str(image_path),
        "mask_count": len(masks),
        "height": int(img_rgb.shape[0]),
        "width": int(img_rgb.shape[1]),
        "r_mean": float(np.mean(img_rgb[:, :, 0])),
    }
    if masks:
        for field in masks[0].keys():
            entry[field] = np.stack([m[field] for m in masks])
    return entry


@contextmanager
def _open_source(source: Any, mask_generator: Any = None):
    """Yield ``(iter_entries, get_entry, raw_handle)`` for an H5, dict, or image source."""
    if source is None:
        if mask_generator is None:
            raise ValueError("A source (.h5 path, open H5 file, dict, or image path(s)) is required.")
        yield (lambda: iter(())), (lambda k: _entry_from_image(k, mask_generator)), None
        return

    if isinstance(source, (str, Path)) and Path(source).suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(source, "r") as h5:
            with _open_source(h5, mask_generator) as res:
                yield res
        return

    if isinstance(source, (h5py.File, h5py.Group)):
        path_map: dict[str, str] | None = None

        def _iter_h5():
            for k in sorted(source.keys()):
                if not k.startswith("temp_"):
                    yield k, H5Entry(source[k], k)

        def _get_h5(key: str) -> H5Entry:
            nonlocal path_map
            if (
                key
                and "/" not in key
                and not key.startswith("temp_")
                and key in source
                and isinstance(source[key], h5py.Group)
            ):
                return H5Entry(source[key], key)
            if path_map is None:
                path_map = {}
                for gk in source.keys():
                    if gk.startswith("temp_"):
                        continue
                    im_p = source[gk].attrs.get("image_path")
                    if isinstance(im_p, bytes):
                        im_p = im_p.decode("utf-8")
                    if im_p:
                        path_map[str(im_p)] = gk
                        path_map.setdefault(os.path.basename(str(im_p)), gk)
            gk = path_map.get(key) or path_map.get(os.path.basename(key))
            if gk is not None:
                return H5Entry(source[gk], gk)
            raise KeyError(f"Key {key!r} not found in HDF5 source")

        yield _iter_h5, _get_h5, source
    elif isinstance(source, Mapping):
        yield (
            lambda: ((str(k), _normalize_dict_entry(v, str(k))) for k, v in source.items()),
            lambda k: _normalize_dict_entry(source[k], str(k)),
            source,
        )
    elif isinstance(source, (str, Path, Sequence)):
        if isinstance(source, (str, Path)):
            p = Path(source)
            paths = get_image_paths(p) if p.is_dir() else [str(p)]
        else:
            paths = [str(p) for p in source]
        yield (
            lambda: ((im, _entry_from_image(im, mask_generator)) for im in paths),
            lambda k: _entry_from_image(k, mask_generator),
            source,
        )
    else:
        raise TypeError(f"Unsupported source type: {type(source).__name__}")


# ---------------------------------------------------------------------------
# Core Port & Adapters
# ---------------------------------------------------------------------------


def build_prefix_maps(index: Mapping[str, Sequence[Any]]) -> dict[int, dict[str, list[Any]]]:
    """Build {prefix_len: {prefix_bits: [keys...]}} from an inverted index."""
    prefix_maps: dict[int, dict[str, list[Any]]] = {}
    for binary_seq, items in index.items():
        for prefix_len in range(1, len(binary_seq) + 1):
            prefix = binary_seq[:prefix_len]
            bucket = prefix_maps.setdefault(prefix_len, {}).setdefault(prefix, [])
            bucket.extend(items)
    return prefix_maps


class CoverlessStegPort(ABC):
    """Base port for coverless steganography adapters.

    Subclasses only need to implement ``encode_entry(entry) -> str``.
    """

    build_prefix_maps = staticmethod(build_prefix_maps)

    def __init__(self, source: Any = None, mask_generator: Any = None):
        if (
            mask_generator is None
            and source is not None
            and callable(getattr(source, "generate", None))
        ):
            mask_generator, source = source, None
        self.source = source
        self.mask_generator = mask_generator
        self._prefix_maps: dict[int, dict[str, list[Any]]] | None = None
        self._cache_key: tuple[Any, ...] | None = None

    def _compute_cache_key(self) -> tuple[Any, ...]:
        def _val_sig(x: Any) -> Any:
            return x if isinstance(x, (int, float, str, bool, bytes, Path, type(None))) else id(x)

        src = self.source
        if isinstance(src, dict):
            src_sig: Any = (
                id(src),
                tuple(
                    (
                        k,
                        tuple((ik, _val_sig(iv)) for ik, iv in sorted(v.items()))
                        if isinstance(v, dict)
                        else _val_sig(v),
                    )
                    for k, v in src.items()
                ),
            )
        else:
            src_sig = _val_sig(src)
        cfg_sig = tuple(
            (k, _val_sig(v))
            for k, v in sorted(self.__dict__.items())
            if not k.startswith("_") and k != "source"
        )
        return (src_sig, cfg_sig)

    @abstractmethod
    def encode_entry(self, entry: Mapping[str, Any]) -> str:
        """Convert one mask/image entry into a binary bitstring ('0'/'1')."""

    def build_index(self, source: Any = None) -> dict[str, list[str]]:
        """Encode every entry in *source* into {bitstring: [entry_keys]}."""
        target = self.source if source is None else source
        index: dict[str, list[str]] = {}
        with _open_source(target, self.mask_generator) as (iter_entries, _, _):
            for key, entry in iter_entries():
                seq = self.encode_entry(entry)
                if seq:
                    index.setdefault(seq, []).append(key)
        return index

    def embed(
        self,
        binary_msg: str,
        max_window_size: int | None = None,
        prefix_maps: Mapping[int, Mapping[str, Sequence[Any]]] | None = None,
    ) -> tuple[list[str], int, int]:
        """Match *binary_msg* chunks against *prefix_maps*, stepping down window size.

        Returns ``(matched_keys, window_size, num_zeros)``, or ``([], -1, -1)`` on failure.
        """
        if any(ch not in ("0", "1") for ch in binary_msg):
            raise ValueError("binary_msg must contain only '0' and '1'.")
        if binary_msg == "":
            return [], 0, 0

        if prefix_maps is None:
            ck = self._compute_cache_key()
            if self._prefix_maps is None or self._cache_key != ck:
                self._prefix_maps = build_prefix_maps(self.build_index(self.source))
                self._cache_key = ck
            prefix_maps = self._prefix_maps

        if not prefix_maps:
            return [], -1, -1

        max_available = max(prefix_maps.keys())
        upper = max_available if max_window_size is None else min(max_window_size, max_available)

        for w in range(upper, 0, -1):
            pmap = prefix_maps.get(w)
            if not pmap:
                continue

            matched_keys: list[str] = []
            num_zeros = 0
            success = True

            for i in range(0, len(binary_msg), w):
                chunk = binary_msg[i : i + w]
                if len(chunk) < w:
                    num_zeros = w - len(chunk)
                    chunk = chunk.ljust(w, "0")
                else:
                    num_zeros = 0

                candidates = pmap.get(chunk)
                if not candidates:
                    success = False
                    break

                first = candidates[0]
                matched_keys.append(
                    first["image path"] if isinstance(first, Mapping) else str(first)
                )

            if success:
                return matched_keys, w, num_zeros

        return [], -1, -1

    def extract(
        self,
        matched_keys: Sequence[str],
        window_size: int,
        num_zeros: int = 0,
        source: Any = None,
    ) -> str:
        """Re-encode *matched_keys* in order from *source* and strip *num_zeros* padding."""
        if not matched_keys or window_size <= 0:
            return ""
        if num_zeros < 0:
            raise ValueError(f"num_zeros must be >= 0, got {num_zeros}")

        target = self.source if source is None else source
        chunks: list[str] = []
        with _open_source(target, self.mask_generator) as (_, get_entry, _):
            for key in matched_keys:
                seq = self.encode_entry(get_entry(key)) or ""
                if len(seq) < window_size:
                    warnings.warn(
                        f"Extracted chunk for {key!r} has length {len(seq)} < window_size={window_size}; "
                        "right-padding with '0'.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    chunks.append(seq.ljust(window_size, "0"))
                else:
                    chunks.append(seq[:window_size])

        bin_msg = "".join(chunks)
        if num_zeros > 0:
            bin_msg = bin_msg[:-num_zeros] if num_zeros < len(bin_msg) else ""
        return bin_msg


class SAMAdapter(CoverlessStegPort):
    """Grid-neighbor SAM mask adapter supporting precomputed H5/dict or raw images."""

    def __init__(
        self,
        source: Any = None,
        grid_size: int = 16,
        tolerance: float = 0.02,
        mode: str = "count",
        mask_generator: Any = None,
    ):
        if mode not in ("count", "binary"):
            raise ValueError(f"mode must be 'count' or 'binary', got {mode!r}")
        if grid_size < 2:
            raise ValueError(f"grid_size must be >= 2, got {grid_size}")
        super().__init__(source=source, mask_generator=mask_generator)
        self.grid_size = grid_size
        self.tolerance = tolerance
        self.mode = mode

    def encode_entry(self, entry: Mapping[str, Any]) -> str:
        seg = entry.get("segmentation")
        if seg is None or len(seg) == 0:
            features = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        else:
            h, w = int(entry.get("height", 0)), int(entry.get("width", 0))
            count_mat, bin_mat = compute_mask_matrices(seg, h, w)
            mat = count_mat if self.mode == "count" else bin_mat
            features = calculate_grid_means(mat, self.grid_size)
        return compare_neighbors(features, self.tolerance)


class BBoxDistanceAdapter(CoverlessStegPort):
    """Lightweight adapter encoding pairwise Euclidean distances between mask bboxes."""

    def __init__(self, source: Any = None, threshold: float = 300.0, mask_generator: Any = None):
        super().__init__(source=source, mask_generator=mask_generator)
        self.threshold = threshold

    def encode_entry(self, entry: Mapping[str, Any]) -> str:
        bboxes = entry.get("bbox")
        if bboxes is None or len(bboxes) < 2:
            return ""
        pts = np.asarray(bboxes, dtype=np.float64)[:, :2]
        dists = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        return "".join("0" if d <= self.threshold else "1" for d in dists)


# ---------------------------------------------------------------------------
# HDF5 Dataset Generator (Colab GPU friendly, resume-safe, zero temp files)
# ---------------------------------------------------------------------------


def generate_sam_h5(
    output_file: str | Path,
    images_path: str | Path | Sequence[str],
    mask_generator: Any,
    *,
    with_noise: bool = False,
    with_compression: bool = False,
    noise_mean: float = 0.0,
    noise_variance: float = 0.001,
    jpeg_quality: int = 90,
    num_images: int = 500,
    seed: int = 42,
) -> Path:
    """Run SAM over *images_path* and write compressed mask outputs to an HDF5 file."""
    if isinstance(images_path, (str, Path)):
        paths = get_image_paths(images_path)[:num_images]
    else:
        paths = [str(p) for p in list(images_path)[:num_images]]
    if not paths:
        raise FileNotFoundError(f"No images found in {images_path!r}")

    out = Path(output_file)
    if str(output_file).endswith(("/", os.sep)) or out.is_dir():
        if with_noise and with_compression:
            tag = "noise_compression"
        elif with_noise:
            tag = "noise"
        elif with_compression:
            tag = "compression"
        else:
            tag = "clean"
        out = out / f"SAM_outputs_{tag}.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    existed_before = out.exists()

    try:
        import torch  # type: ignore

        inference_ctx = torch.inference_mode
    except ImportError:
        inference_ctx = nullcontext

    def _load_rgb(im_path: str) -> np.ndarray | None:
        bgr = cv2.imread(im_path)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    try:
        with h5py.File(
            out, "a", rdcc_nbytes=64 * 1024 * 1024, rdcc_nslots=10007, rdcc_w0=1.0
        ) as h5:
            existing_keys = set(h5.keys())
            pending = [
                (i, f"{i:04d}", im)
                for i, im in enumerate(paths)
                if f"{i:04d}" not in existing_keys
            ]
            if not pending:
                return out

            def _write_entry(
                key: str,
                im_path: str,
                height: int,
                width: int,
                r_mean: float,
                outputs: list[dict[str, Any]],
            ) -> None:
                temp_key = f"temp_{key}"
                if temp_key in h5:
                    del h5[temp_key]
                group = h5.create_group(temp_key)
                group.attrs["image_path"] = im_path
                group.attrs["mask_count"] = len(outputs)
                group.attrs["height"] = height
                group.attrs["width"] = width
                group.attrs["r_mean"] = r_mean
                if outputs:
                    for field in outputs[0].keys():
                        values = np.stack([m[field] for m in outputs])
                        group.create_dataset(
                            field, data=values, compression="gzip", compression_opts=4
                        )
                h5.move(temp_key, key)

            prefetch_limit = 8
            max_queued_writes = 2
            pending_iter = iter(pending)
            read_futures: deque[tuple[int, str, str, Any]] = deque()
            write_futures: deque[Any] = deque()

            with (
                ThreadPoolExecutor(max_workers=4) as reader_pool,
                ThreadPoolExecutor(max_workers=1) as writer_pool,
                inference_ctx(),
            ):
                try:
                    for _ in range(prefetch_limit):
                        nxt = next(pending_iter, None)
                        if nxt is None:
                            break
                        idx, k, im = nxt
                        read_futures.append((idx, k, im, reader_pool.submit(_load_rgb, im)))

                    while read_futures:
                        idx, key, im, fut = read_futures.popleft()
                        nxt = next(pending_iter, None)
                        if nxt is not None:
                            nidx, nk, nim = nxt
                            read_futures.append(
                                (nidx, nk, nim, reader_pool.submit(_load_rgb, nim))
                            )

                        image = fut.result()
                        if image is None:
                            raise ValueError(f"Could not read image: {im}")

                        if with_noise:
                            image = apply_gaussian_noise(
                                image, mean=noise_mean, variance=noise_variance, rng=seed + idx
                            )
                        if with_compression:
                            image = apply_jpeg_comp(image, quality=jpeg_quality)

                        h_px, w_px = int(image.shape[0]), int(image.shape[1])
                        r_mean = float(np.mean(image[:, :, 0]))
                        outputs = mask_generator.generate(image)
                        del image

                        if len(write_futures) >= max_queued_writes:
                            write_futures.popleft().result()
                        write_futures.append(
                            writer_pool.submit(_write_entry, key, im, h_px, w_px, r_mean, outputs)
                        )

                    while write_futures:
                        write_futures.popleft().result()
                    h5.flush()
                finally:
                    for *_, f in read_futures:
                        f.cancel()
    except Exception:
        if not existed_before and out.exists():
            with h5py.File(out, "r") as check_h5:
                empty = len(check_h5.keys()) == 0
            if empty:
                out.unlink()
        raise

    return out


# ---------------------------------------------------------------------------
# Adapter Variance Tester (embed input vs extract output discrepancy)
# ---------------------------------------------------------------------------


def compute_bit_errors(expected_bits: str, extracted_bits: str) -> tuple[int, float]:
    """Return ``(bit_errors, ber)`` counting length mismatches as bit errors."""
    bit_errors = sum(a != b for a, b in zip(expected_bits, extracted_bits)) + abs(
        len(expected_bits) - len(extracted_bits)
    )
    if len(expected_bits) == 0:
        ber = 0.0 if len(extracted_bits) == 0 else 1.0
    else:
        ber = bit_errors / len(expected_bits)
    return bit_errors, ber


@dataclass
class MessageResult:
    message: str
    binary_msg: str
    extracted_bin_msg: str
    embed_succeeded: bool
    matched_keys: list[str]
    window_size: int
    num_zeros: int
    images_used: int
    hamming_distance: int
    ber: float
    exact_match: bool


@dataclass
class ConditionSummary:
    condition: str
    num_messages: int
    mean_ber: float
    ber_variance: float
    ber_std: float
    mean_hamming_distance: float
    hamming_variance: float
    exact_match_rate: float
    embed_failure_rate: float
    avg_window_size: float
    avg_images_used: float
    elapsed_sec: float
    messages: list[MessageResult]


@dataclass
class VarianceReport:
    adapter_name: str
    index_keys: int
    index_build_sec: float
    embed_failure_rate: float
    conditions: dict[str, ConditionSummary]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_table(self) -> str:
        header = (
            f"Adapter: {self.adapter_name} | Index keys: {self.index_keys} "
            f"({self.index_build_sec:.3f}s) | Embed failure rate: {self.embed_failure_rate:.1%}\n"
            f"{'Condition':<14} {'Mean BER':>10} {'BER Var':>10} {'BER Std':>10} "
            f"{'Mean Ham':>10} {'Ham Var':>10} {'Exact':>8} {'Avg Win':>8} {'Avg Imgs':>9} {'Time(s)':>8}"
        )
        lines = [header, "-" * len(header.splitlines()[-1])]
        for cond, s in self.conditions.items():
            lines.append(
                f"{cond:<14} {s.mean_ber:>10.4f} {s.ber_variance:>10.6f} {s.ber_std:>10.4f} "
                f"{s.mean_hamming_distance:>10.2f} {s.hamming_variance:>10.2f} "
                f"{s.exact_match_rate:>7.1%} {s.avg_window_size:>8.2f} "
                f"{s.avg_images_used:>9.2f} {s.elapsed_sec:>8.3f}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format_table()


def test_adapter_variance(
    adapter: CoverlessStegPort,
    clean_source: Any,
    extraction_sources: Mapping[str, Any] | Any = None,
    messages: Sequence[str] = ("HelloWorld", "Coverless", "SAM Steg"),
    *,
    is_binary: bool | None = None,
    max_window_size: int | None = None,
) -> VarianceReport:
    """Measure bit-level variance between messages into ``embed`` and out of ``extract``."""
    if extraction_sources is None:
        sources_map: Mapping[str, Any] = {"clean": clean_source}
    elif isinstance(extraction_sources, (str, Path, h5py.File, h5py.Group)) or not isinstance(
        extraction_sources, Mapping
    ):
        sources_map = {"default": extraction_sources}
    else:
        sources_map = extraction_sources

    t_idx = time.perf_counter()
    index = adapter.build_index(clean_source)
    prefix_maps = adapter.build_prefix_maps(index)
    index_build_sec = time.perf_counter() - t_idx

    prepared: list[tuple[str, str, list[str], int, int, bool]] = []
    for msg in messages:
        treat_as_bin = (
            is_binary
            if is_binary is not None
            else (len(msg) > 0 and all(ch in ("0", "1") for ch in msg))
        )
        bin_msg = msg if treat_as_bin else string_to_binary(msg)
        matched_keys, w_size, num_zeros = adapter.embed(
            bin_msg, max_window_size=max_window_size, prefix_maps=prefix_maps
        )
        succeeded = w_size >= 0
        prepared.append((msg, bin_msg, matched_keys, w_size, num_zeros, succeeded))

    total = len(prepared)
    embed_failures = sum(1 for *_, ok in prepared if not ok)
    embed_failure_rate = (embed_failures / total) if total > 0 else 0.0
    succeeded_wins = [w for _, _, _, w, _, ok in prepared if ok]
    succeeded_imgs = [len(k) for _, _, k, _, _, ok in prepared if ok]
    avg_win = float(np.mean(succeeded_wins)) if succeeded_wins else 0.0
    avg_imgs = float(np.mean(succeeded_imgs)) if succeeded_imgs else 0.0

    conditions: dict[str, ConditionSummary] = {}
    for cond_name, target_source in sources_map.items():
        t0 = time.perf_counter()
        msg_results: list[MessageResult] = []

        # Keep HDF5 file open across all messages in a condition rather than reopening per message.
        with _open_source(target_source, adapter.mask_generator) as (_, _, open_ref):
            for raw_msg, bin_msg, matched_keys, w_size, num_zeros, ok in prepared:
                if ok:
                    ext_bits = adapter.extract(
                        matched_keys, w_size, num_zeros=num_zeros, source=open_ref
                    )
                else:
                    ext_bits = ""
                bit_err, ber = compute_bit_errors(bin_msg, ext_bits)
                exact = ok and (ext_bits == bin_msg)
                msg_results.append(
                    MessageResult(
                        message=raw_msg,
                        binary_msg=bin_msg,
                        extracted_bin_msg=ext_bits,
                        embed_succeeded=ok,
                        matched_keys=list(matched_keys),
                        window_size=w_size,
                        num_zeros=num_zeros,
                        images_used=len(matched_keys),
                        hamming_distance=bit_err,
                        ber=ber,
                        exact_match=exact,
                    )
                )

        elapsed = time.perf_counter() - t0
        bers = np.array([r.ber for r in msg_results], dtype=np.float64)
        hams = np.array([r.hamming_distance for r in msg_results], dtype=np.float64)
        exacts = np.array([1.0 if r.exact_match else 0.0 for r in msg_results], dtype=np.float64)
        conditions[cond_name] = ConditionSummary(
            condition=cond_name,
            num_messages=total,
            mean_ber=float(np.mean(bers)) if total else 0.0,
            ber_variance=float(np.var(bers)) if total else 0.0,
            ber_std=float(np.std(bers)) if total else 0.0,
            mean_hamming_distance=float(np.mean(hams)) if total else 0.0,
            hamming_variance=float(np.var(hams)) if total else 0.0,
            exact_match_rate=float(np.mean(exacts)) if total else 0.0,
            embed_failure_rate=embed_failure_rate,
            avg_window_size=avg_win,
            avg_images_used=avg_imgs,
            elapsed_sec=elapsed,
            messages=msg_results,
        )

    return VarianceReport(
        adapter_name=type(adapter).__name__,
        index_keys=len(index),
        index_build_sec=index_build_sec,
        embed_failure_rate=embed_failure_rate,
        conditions=conditions,
    )


test_adapter_variance.__test__ = False  # type: ignore[attr-defined]

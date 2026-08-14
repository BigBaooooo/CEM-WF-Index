"""Read-only access to precomputed ERA5 background-context fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


_VIEW_HOURS = {"1h": 1, "hourly": 1, "6h": 6, "six_hour": 6, "daily": 24, "24h": 24}


def _datetime64_ns(value: Any) -> np.datetime64:
    timestamp = np.datetime64(value, "ns")
    if np.isnat(timestamp):
        raise ValueError(f"Invalid timestamp: {value!r}")
    return timestamp


@dataclass(frozen=True)
class FingerprintView:
    """A lightweight strided view over a :class:`FingerprintArchive`."""

    archive: "FingerprintArchive"
    row_indices: np.ndarray
    stride_hours: int

    def __len__(self) -> int:
        return int(len(self.row_indices))

    @property
    def start_times(self) -> np.ndarray:
        return self.archive.start_times[self.row_indices]

    @property
    def end_times(self) -> np.ndarray:
        return self.archive.end_times[self.row_indices]

    def vectors(self, rows: slice | np.ndarray | list[int] | None = None) -> np.ndarray:
        """Materialize selected vectors while leaving the archive memory-mapped."""

        indices = self.row_indices if rows is None else self.row_indices[rows]
        return np.asarray(self.archive.matrix[indices], dtype=np.float32)


class FingerprintArchive:
    """Validated fingerprint matrix with aligned weather-window timestamps.

    The three arrays remain read-only.  Structural validation is always run;
    the optional finite-value scan is chunked so a large memory-mapped matrix
    does not need to be copied into memory.
    """

    def __init__(
        self,
        matrix_path: str | Path,
        start_times_path: str | Path,
        end_times_path: str | Path,
        *,
        expected_dim: int = 1024,
        mmap: bool = True,
    ) -> None:
        self.matrix_path = Path(matrix_path)
        self.start_times_path = Path(start_times_path)
        self.end_times_path = Path(end_times_path)
        for path in (self.matrix_path, self.start_times_path, self.end_times_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        mmap_mode = "r" if mmap else None
        self.matrix = np.load(self.matrix_path, mmap_mode=mmap_mode, allow_pickle=False)
        self.start_times = np.load(self.start_times_path, mmap_mode=mmap_mode, allow_pickle=False)
        self.end_times = np.load(self.end_times_path, mmap_mode=mmap_mode, allow_pickle=False)
        self.expected_dim = int(expected_dim)
        self._validate_structure()

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        expected_dim: int = 1024,
        mmap: bool = True,
    ) -> "FingerprintArchive":
        root = Path(directory)
        return cls(
            root / "all_fp1024.npy",
            root / "start_times.npy",
            root / "end_times.npy",
            expected_dim=expected_dim,
            mmap=mmap,
        )

    def _validate_structure(self) -> None:
        if self.matrix.ndim != 2 or self.matrix.shape[1] != self.expected_dim:
            raise ValueError(
                f"Fingerprint matrix must have shape (N, {self.expected_dim}); got {self.matrix.shape}"
            )
        if not np.issubdtype(self.matrix.dtype, np.floating):
            raise TypeError(f"Fingerprint matrix must be floating point; got {self.matrix.dtype}")
        if self.start_times.ndim != 1 or self.end_times.ndim != 1:
            raise ValueError("start_times and end_times must be one-dimensional")
        if len(self.matrix) != len(self.start_times) or len(self.matrix) != len(self.end_times):
            raise ValueError("Fingerprint rows and timestamp arrays must have identical lengths")
        if not np.issubdtype(self.start_times.dtype, np.datetime64) or not np.issubdtype(
            self.end_times.dtype, np.datetime64
        ):
            raise TypeError("Timestamp arrays must use a NumPy datetime64 dtype")
        if len(self.matrix) == 0:
            raise ValueError("Fingerprint archive is empty")

        starts = self.start_times.astype("datetime64[ns]")
        ends = self.end_times.astype("datetime64[ns]")
        if np.isnat(starts).any() or np.isnat(ends).any():
            raise ValueError("Timestamp arrays contain NaT")
        if np.any(ends <= starts):
            raise ValueError("Every fingerprint window must end after it starts")
        if len(starts) > 1 and np.any(starts[1:] <= starts[:-1]):
            raise ValueError("Fingerprint start times must be strictly increasing")

    def close(self) -> None:
        """Close any NumPy memory maps held by this archive."""

        for array in (self.matrix, self.start_times, self.end_times):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def __enter__(self) -> "FingerprintArchive":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[1])

    def validate(self, *, check_finite: bool = True, batch_size: int = 16_384) -> dict[str, Any]:
        """Return a compact validation receipt without exposing local paths."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        finite = True
        if check_finite:
            for start in range(0, len(self.matrix), batch_size):
                if not np.isfinite(np.asarray(self.matrix[start : start + batch_size])).all():
                    finite = False
                    break
        return {
            "row_count": int(len(self.matrix)),
            "dimension": self.dimension,
            "dtype": str(self.matrix.dtype),
            "time_coverage_start": str(self.start_times[0].astype("datetime64[s]")),
            "time_coverage_end": str(self.end_times[-1].astype("datetime64[s]")),
            "strictly_increasing_start_times": True,
            "finite_values": finite if check_finite else "not_scanned",
        }

    def view(self, cadence: str | int = "1h") -> FingerprintView:
        """Return the rows aligned to an hourly, six-hourly, or daily cadence."""

        if isinstance(cadence, str):
            try:
                stride_hours = _VIEW_HOURS[cadence.lower()]
            except KeyError as exc:
                raise ValueError(f"Unsupported fingerprint cadence: {cadence!r}") from exc
        else:
            stride_hours = int(cadence)
        if stride_hours <= 0:
            raise ValueError("cadence must be a positive number of hours")

        starts = self.start_times.astype("datetime64[ns]")
        elapsed_ns = (starts - starts[0]).astype("timedelta64[ns]").astype(np.int64)
        hour_ns = int(np.timedelta64(1, "h").astype("timedelta64[ns]").astype(np.int64))
        if np.any(elapsed_ns % hour_ns != 0):
            raise ValueError("Fingerprint start times are not aligned to whole hours")
        elapsed_hours = elapsed_ns // hour_ns
        indices = np.flatnonzero(elapsed_hours % stride_hours == 0).astype(np.int64)
        return FingerprintView(self, indices, stride_hours)

    def rows_for_event(
        self,
        event_start: Any,
        event_end: Any,
        *,
        cadence: str | int = "1h",
    ) -> np.ndarray:
        """Return rows whose weather-window start lies in ``[start, end)``."""

        start = _datetime64_ns(event_start)
        end = _datetime64_ns(event_end)
        if end <= start:
            raise ValueError("event_end must be after event_start")
        view = self.view(cadence)
        starts = view.start_times.astype("datetime64[ns]")
        local = np.flatnonzero((starts >= start) & (starts < end))
        if len(local) == 0:
            raise ValueError("No fingerprint rows cover the requested event interval")
        return view.row_indices[local]

    def normalized_mean_pool(
        self,
        event_start: Any,
        event_end: Any,
        *,
        cadence: str | int = "1h",
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        unit_norm: bool = True,
    ) -> np.ndarray:
        """Mean-pool an event window after optional fixed feature normalization.

        ``feature_mean`` and ``feature_std`` must be frozen external statistics
        (for example, statistics estimated on training data).  This method never
        estimates normalization statistics from a query or evaluation split.
        """

        rows = self.rows_for_event(event_start, event_end, cadence=cadence)
        values = np.asarray(self.matrix[rows], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Selected fingerprint rows contain non-finite values")
        if (feature_mean is None) != (feature_std is None):
            raise ValueError("feature_mean and feature_std must be supplied together")
        if feature_mean is not None and feature_std is not None:
            mean = np.asarray(feature_mean, dtype=np.float32).reshape(-1)
            std = np.asarray(feature_std, dtype=np.float32).reshape(-1)
            if mean.shape != (self.dimension,) or std.shape != (self.dimension,):
                raise ValueError("Normalization vectors must match the fingerprint dimension")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("Normalization statistics must be finite with strictly positive std")
            values = (values - mean) / std
        pooled = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        if unit_norm:
            norm = float(np.linalg.norm(pooled))
            if norm > 1e-12:
                pooled = pooled / norm
        return pooled.astype(np.float32, copy=False)

    def iter_batches(self, *, batch_size: int = 16_384) -> Iterator[np.ndarray]:
        """Yield read-only-sized batches for index construction."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self.matrix), batch_size):
            yield np.asarray(self.matrix[start : start + batch_size], dtype=np.float32)

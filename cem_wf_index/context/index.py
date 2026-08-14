"""L2 context index with an optional HNSW backend and exact fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SearchHit:
    candidate_id: str
    rank: int
    distance: float
    provenance: str = "background_context_index"
    group_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextIndex:
    """Index 1,024-dimensional context vectors under squared L2 distance.

    ``backend="auto"`` uses ``hnswlib`` when it is available and otherwise
    selects the deterministic exact implementation.  Both backends return the
    same distance definition and output schema.
    """

    def __init__(
        self,
        *,
        dimension: int = 1024,
        backend: str = "auto",
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 200,
        random_seed: int = 20262202,
    ) -> None:
        if dimension <= 0 or m <= 0 or ef_construction <= 0 or ef_search <= 0:
            raise ValueError("ContextIndex dimensions and HNSW parameters must be positive")
        if backend not in {"auto", "hnsw", "exact"}:
            raise ValueError("backend must be 'auto', 'hnsw', or 'exact'")
        self.dimension = int(dimension)
        self.backend = backend
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.random_seed = int(random_seed)
        self.backend_used: str | None = None
        self._vectors: np.ndarray | None = None
        self._ids: np.ndarray | None = None
        self._groups: np.ndarray | None = None
        self._hnsw: Any = None

    @staticmethod
    def _hnswlib() -> Any | None:
        try:
            import hnswlib  # type: ignore[import-not-found]
        except ImportError:
            return None
        return hnswlib

    def fit(
        self,
        vectors: np.ndarray,
        candidate_ids: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
    ) -> "ContextIndex":
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError(f"vectors must have shape (N, {self.dimension}); got {matrix.shape}")
        if len(matrix) == 0:
            raise ValueError("Cannot build an empty ContextIndex")
        if not np.isfinite(matrix).all():
            raise ValueError("Context vectors contain non-finite values")
        ids = np.asarray([str(value) for value in candidate_ids], dtype=object)
        if len(ids) != len(matrix):
            raise ValueError("candidate_ids must align one-to-one with vectors")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("candidate_ids must be unique")
        if group_ids is None:
            groups = np.full(len(ids), None, dtype=object)
        else:
            if len(group_ids) != len(ids):
                raise ValueError("group_ids must align one-to-one with vectors")
            groups = np.asarray([None if value is None else str(value) for value in group_ids], dtype=object)

        hnswlib = self._hnswlib()
        if self.backend == "hnsw" and hnswlib is None:
            raise RuntimeError("backend='hnsw' requires the optional hnswlib package")
        use_hnsw = self.backend == "hnsw" or (self.backend == "auto" and hnswlib is not None)
        self._vectors = matrix
        self._ids = ids
        self._groups = groups
        if use_hnsw:
            index = hnswlib.Index(space="l2", dim=self.dimension)
            index.init_index(
                max_elements=len(matrix),
                ef_construction=self.ef_construction,
                M=self.m,
                random_seed=self.random_seed,
            )
            index.add_items(matrix, np.arange(len(matrix), dtype=np.int64), num_threads=1)
            index.set_ef(max(self.ef_search, 1))
            index.set_num_threads(1)
            self._hnsw = index
            self.backend_used = "hnsw"
        else:
            self._hnsw = None
            self.backend_used = "exact"
        return self

    def __len__(self) -> int:
        return 0 if self._ids is None else int(len(self._ids))

    def _check_ready(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._vectors is None or self._ids is None or self._groups is None or self.backend_used is None:
            raise RuntimeError("ContextIndex.fit must be called before search")
        return self._vectors, self._ids, self._groups

    def _ordered_positions(self, query: np.ndarray, requested: int) -> tuple[np.ndarray, np.ndarray]:
        vectors, ids, _ = self._check_ready()
        requested = min(max(int(requested), 1), len(vectors))
        if self.backend_used == "hnsw":
            labels, distances = self._hnsw.knn_query(query.reshape(1, -1), k=requested, num_threads=1)
            positions = labels[0].astype(np.int64)
            values = distances[0].astype(np.float64)
        else:
            values_all = np.einsum("ij,ij->i", vectors - query, vectors - query, dtype=np.float64)
            positions = np.arange(len(vectors), dtype=np.int64)
            order = sorted(range(len(vectors)), key=lambda pos: (float(values_all[pos]), str(ids[pos])))[:requested]
            positions = positions[np.asarray(order, dtype=np.int64)]
            values = values_all[positions]
        order = sorted(range(len(positions)), key=lambda pos: (float(values[pos]), str(ids[positions[pos]])))
        order_array = np.asarray(order, dtype=np.int64)
        return positions[order_array], values[order_array]

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        *,
        exclude_ids: Iterable[str] = (),
        exclude_group_ids: Iterable[str] = (),
        oversample_factor: int = 3,
    ) -> list[SearchHit]:
        vectors, ids, groups = self._check_ready()
        if k <= 0:
            return []
        if oversample_factor <= 0:
            raise ValueError("oversample_factor must be positive")
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape != (self.dimension,) or not np.isfinite(query).all():
            raise ValueError(f"query_vector must be a finite vector of length {self.dimension}")
        blocked_ids = {str(value) for value in exclude_ids}
        blocked_groups = {str(value) for value in exclude_group_ids}
        eligible = sum(
            1
            for candidate_id, group_id in zip(ids, groups)
            if str(candidate_id) not in blocked_ids
            and (group_id is None or str(group_id) not in blocked_groups)
        )
        target = min(int(k), eligible)
        if target == 0:
            return []

        requested = min(len(vectors), max(target * oversample_factor, target + len(blocked_ids)))
        accepted: list[tuple[int, float]] = []
        while True:
            positions, distances = self._ordered_positions(query, requested)
            accepted = [
                (int(pos), float(distance))
                for pos, distance in zip(positions, distances)
                if str(ids[pos]) not in blocked_ids
                and (groups[pos] is None or str(groups[pos]) not in blocked_groups)
            ]
            if len(accepted) >= target or requested == len(vectors):
                break
            requested = min(len(vectors), max(requested + 1, requested * 2))

        return [
            SearchHit(
                candidate_id=str(ids[pos]),
                rank=rank,
                distance=distance,
                group_id=None if groups[pos] is None else str(groups[pos]),
            )
            for rank, (pos, distance) in enumerate(accepted[:target], start=1)
        ]

    def configuration(self) -> dict[str, Any]:
        return {
            "backend": self.backend_used or self.backend,
            "space": "l2",
            "dimension": self.dimension,
            "M": self.m,
            "efConstruction": self.ef_construction,
            "efSearch": self.ef_search,
        }

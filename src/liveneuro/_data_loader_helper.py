"""
Data Loader Helper - Responsible for all data ingestion logic.

This helper's single responsibility is to handle data loading - reading NDVar,
MNE SourceEstimate, or sample data and normalizing it into a consistent
internal format.
Loading and normalization are both part of one cohesive responsibility:
preparing data for visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
from eelbrain import NDVar, datasets


@dataclass
class BrainData:
    """Container for brain data and basic metadata."""

    glass_brain_data: np.ndarray  # Always (n_sources, 3, n_times)
    butterfly_data: np.ndarray  # Always (n_sources, n_times)
    source_coords: np.ndarray  # (n_sources, 3)
    time_values: np.ndarray  # (n_times,)


class DataLoaderHelper:
    """Helper responsible for data ingestion and normalization.

    This helper has a single responsibility: preparing data for visualization.
    It handles:
    - Loading NDVar data directly
    - Loading MNE volume vector source estimates
    - Loading MNE sample data
    - Normalizing data into a consistent internal format
    - Computing derived data (butterfly data from vector norms)
    """

    @staticmethod
    def load_source_data() -> BrainData:
        """Load MNE sample data and prepare for 2D brain visualization.

        Returns
        -------
        BrainData
            Loaded brain data and metadata.
        """
        # Load MNE sample data
        data_ds = datasets.get_mne_sample(src="vol", ori="vector")

        # Average over trials/cases
        src_ndvar = data_ds["src"].mean("case")

        # Extract coordinates and data
        glass_brain_data = src_ndvar.get_data(("source", "space", "time"))
        source_coords = src_ndvar.source.coordinates  # (n_sources, 3)
        time_values = src_ndvar.time.times

        # Compute norm for butterfly plot
        butterfly_data = np.linalg.norm(glass_brain_data, axis=1)

        return BrainData(
            glass_brain_data=glass_brain_data,
            butterfly_data=butterfly_data,
            source_coords=source_coords,
            time_values=time_values,
        )

    @staticmethod
    def load_ndvar_data(y: NDVar) -> BrainData:
        """Load data from NDVar directly.

        Parameters
        ----------
        y
            Data with dimensions ([case,] time, source[, space]).
        Returns
        -------
        BrainData
            Loaded brain data and metadata.
        """
        if y.has_case:
            y = y.mean("case")

        # Extract source dimension info
        source = y.get_dim("source")
        source_coords = source.coordinates
        time_values = y.time.times

        # Handle space dimension (vector data vs scalar data)
        if y.has_dim("space"):
            # Extract 3D vector data (n_sources, 3, n_times)
            glass_brain_data = y.get_data(("source", "space", "time"))
            # Compute norm for butterfly plot (n_sources, n_times)
            butterfly_data = np.linalg.norm(glass_brain_data, axis=1)
        else:
            # Scalar data - no space dimension
            data_2d = y.get_data(("source", "time"))  # (n_sources, n_times)
            butterfly_data = data_2d.copy()
            # Expand to 3D for consistency (assuming scalar represents magnitude)
            glass_brain_data = data_2d[:, np.newaxis, :]  # (n_sources, 1, n_times)

        return BrainData(
            glass_brain_data=glass_brain_data,
            butterfly_data=butterfly_data,
            source_coords=source_coords,
            time_values=time_values,
        )

    @staticmethod
    def load_mne_vol_vector_source_estimate(
        stc: mne.VolVectorSourceEstimate,
        src: mne.SourceSpaces | None,
    ) -> BrainData:
        """Load an MNE VolVectorSourceEstimate with its source space.

        Parameters
        ----------
        stc
            MNE VolVectorSourceEstimate with data shaped ``(n_sources, 3, n_times)``.
        src
            Matching MNE SourceSpaces object. Required because source estimates carry
            vertex ids, while LiveNeuro needs 3D source coordinates for projections.

        Returns
        -------
        BrainData
            Loaded brain data and metadata.
        """
        if src is None:
            raise ValueError(
                "src is required when y is an mne.VolVectorSourceEstimate because "
                "the source estimate does not include 3D source coordinates."
            )

        glass_brain_data = DataLoaderHelper._get_mne_vector_data(stc)
        vertices = DataLoaderHelper._normalize_mne_vertices(stc.vertices)
        source_coords = DataLoaderHelper._source_coords_from_mne_src(vertices, src)
        time_values = np.asarray(stc.times, dtype=float)

        if source_coords.shape[0] != glass_brain_data.shape[0]:
            raise ValueError(
                "MNE source coordinate count does not match source estimate data: "
                f"{source_coords.shape[0]} coordinates for "
                f"{glass_brain_data.shape[0]} data sources."
            )
        if time_values.ndim != 1 or time_values.shape[0] != glass_brain_data.shape[2]:
            raise ValueError(
                "MNE source estimate times must be one-dimensional and match the "
                f"data time axis; got times shape {time_values.shape} and data "
                f"shape {glass_brain_data.shape}."
            )

        butterfly_data = np.linalg.norm(glass_brain_data, axis=1)

        return BrainData(
            glass_brain_data=glass_brain_data,
            butterfly_data=butterfly_data,
            source_coords=source_coords,
            time_values=time_values,
        )

    @staticmethod
    def _get_mne_vector_data(stc: mne.VolVectorSourceEstimate) -> np.ndarray:
        """Extract vector data from an MNE source estimate-like object."""
        candidates = [stc.data]
        private_data = getattr(stc, "_data", None)
        if private_data is not None:
            candidates.append(private_data)

        shapes = []
        for candidate in candidates:
            data = np.asarray(candidate)
            shapes.append(data.shape)
            if data.ndim == 3 and data.shape[1] == 3:
                return data

        raise ValueError(
            "Expected mne.VolVectorSourceEstimate data with shape "
            f"(n_sources, 3, n_times), got {shapes}."
        )

    @staticmethod
    def _normalize_mne_vertices(vertices: list[np.ndarray]) -> list[np.ndarray]:
        """Normalize MNE vertex ids to one array per source space."""
        normalized = [np.asarray(group, dtype=int) for group in vertices]
        if not normalized or any(group.ndim != 1 for group in normalized):
            raise ValueError(
                "MNE VolVectorSourceEstimate vertices must be one-dimensional "
                "arrays, or a list of one-dimensional arrays."
            )
        return normalized

    @staticmethod
    def _source_coords_from_mne_src(
        vertex_groups: list[np.ndarray],
        src: mne.SourceSpaces,
    ) -> np.ndarray:
        """Extract source coordinates from MNE SourceSpaces and STC vertices."""
        if len(vertex_groups) != len(src):
            raise ValueError(
                "MNE source estimate vertices must have one entry per source "
                f"space; got {len(vertex_groups)} vertex group(s) and "
                f"{len(src)} source space(s)."
            )

        coords = [
            DataLoaderHelper._coords_from_mne_source_space(src, i, vertices)
            for i, vertices in enumerate(vertex_groups)
        ]
        return np.concatenate(coords, axis=0)

    @staticmethod
    def _coords_from_mne_source_space(
        src: mne.SourceSpaces, source_space_idx: int, vertices: np.ndarray
    ) -> np.ndarray:
        """Map dense MNE vertex ids to source-space coordinates."""
        space = src[source_space_idx]
        try:
            rr = np.asarray(space["rr"], dtype=float)
        except Exception as exc:
            raise ValueError(
                "Each MNE source space must provide an 'rr' array."
            ) from exc

        if rr.ndim != 2 or rr.shape[1] != 3:
            raise ValueError(
                "MNE source space 'rr' must have shape (n_vertices, 3), got "
                f"{rr.shape}."
            )
        if vertices.size == 0:
            return np.empty((0, 3), dtype=float)
        if np.min(vertices) >= 0 and np.max(vertices) < rr.shape[0]:
            return rr[vertices]

        vertno = None
        try:
            vertno = np.asarray(space.get("vertno"), dtype=int)
        except Exception:
            vertno = None

        if vertno is not None and vertno.ndim == 1 and vertno.shape[0] == rr.shape[0]:
            index_by_vertex = {int(vertex): i for i, vertex in enumerate(vertno)}
            if all(int(vertex) in index_by_vertex for vertex in vertices):
                return rr[[index_by_vertex[int(vertex)] for vertex in vertices]]

        if rr.shape[0] == vertices.shape[0]:
            return rr

        raise ValueError(
            "Could not map MNE source estimate vertices to source-space "
            "coordinates. Pass the matching SourceSpaces object used to create "
            "the VolVectorSourceEstimate."
        )

"""Application-wide configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class AppParameters:
    """
    Parameters for the app orchestrator.

    This dataclass stores configuration values that control import behavior,
    NN and optimization services, serialization defaults, and selected UI
    preferences that need to be persisted between sessions.
    """

    # ---- Core collection parameters ----
    automatic_methods: bool = True
    default_peak_model: str = "pseudo-voigt"
    default_background_model: str = "shirley"

    # ---- UI parameters ----
    show_spectrum_id_in_tree: bool = True
    region_slice_display_mode: Literal["value", "index"] = "value"
    show_id_in_properties_tree: bool = True
    show_residuals_plot: bool = True

    # ---- Import service parameters ----
    import_use_binding_energy: bool = True
    import_use_cps: bool = True

    # ---- NN service parameters ----
    nn_model_path: str | None = "assets/models/model.onnx"
    nn_pred_threshold: float = 0.5
    nn_smooth: bool = True
    nn_interp_num: int = 256

    # ---- Optimization service parameters ----
    optimization_kwargs: dict[str, Any] = field(default_factory=dict)

    # ---- Serialization service parameters ----
    default_serialization_mode: Literal["append", "replace"] = "replace"
    default_serialization_path: str | Path | None = None
    default_serialization_indent: int | None = None
    default_serialization_use_gzip: bool = True
    default_serialization_compresslevel: int = 9

"""
Serialization tools for CoreCollection objects.

Provides functionality to serialize CoreCollection instances to JSON format,
preserving object relationships, metadata, and data in a compact representation.
"""

import gzip
import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from core.collection import CoreCollection
from core.io.post_processing import deserialize_section, serialize_series
from core.math_models import BaseBackgroundModel, BasePeakModel, ModelRegistry
from core.metadata import (
    BackgroundMetadata,
    Metadata,
    PeakMetadata,
    RegionMetadata,
    SpectrumMetadata,
)
from core.objects import Background, Component, Peak, Region, Spectrum
from core.services import MetadataService

LoadMode = Literal["append", "replace", "new"]
DeserializeResult = CoreCollection | tuple[CoreCollection, MetadataService]

VERSION = "1.1"


def _serialize_spectrum(spectrum: Spectrum, metadata: Metadata | None = None) -> dict[str, Any]:
    """
    Serialize a Spectrum object.

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum object to serialize.
    metadata : Metadata, optional
        Metadata associated with the spectrum.

    Returns
    -------
    dict[str, Any]
        Serialized spectrum dictionary.
    """
    x = spectrum.x
    y = spectrum.y

    # Extract x-axis parameters (assuming uniform spacing)
    if len(x) == 0:
        x_start = 0.0
        x_step = 0.0
        x_num_points = 0
    elif len(x) == 1:
        x_start = float(x[0])
        x_step = 0.0
        x_num_points = 1
    else:
        x_start = float(x[0])
        x_step = float((x[-1] - x[0]) / (len(x) - 1))
        x_num_points = len(x)

    # Convert y array to list
    y_list = y.tolist()

    result: dict[str, Any] = {
        "id": spectrum.id_,
        "parent_id": None,
        "type": "Spectrum",
        "x_start": x_start,
        "x_step": x_step,
        "x_num_points": x_num_points,
        "y": y_list,
    }
    # A document can contain several measured energy windows in one spectrum.
    # Keep every energy when the compact regular-grid representation is lossy.
    compact_x = x_start + np.arange(x_num_points) * x_step
    if not np.allclose(x, compact_x, rtol=0, atol=1e-9):
        result["x"] = x.tolist()

    # Add metadata if available
    if metadata is not None:
        result["metadata"] = asdict(metadata)
    else:
        result["metadata"] = None

    return result


def _serialize_region(region: Region, metadata: Metadata | None = None) -> dict[str, Any]:
    """
    Serialize a Region object.

    Parameters
    ----------
    region : Region
        Region object to serialize.
    metadata : Metadata, optional
        Metadata associated with the region.

    Returns
    -------
    dict[str, Any]
        Serialized region dictionary.
    """
    result: dict[str, Any] = {
        "id": region.id_,
        "parent_id": region.parent_id,
        "type": "Region",
        "slice_start": region.slice_.start,
        "slice_stop": region.slice_.stop,
    }

    # Add metadata if available
    if metadata is not None:
        result["metadata"] = asdict(metadata)
    else:
        result["metadata"] = None

    return result


def _serialize_component(component: Component, metadata: Metadata | None = None) -> dict[str, Any]:
    """
    Serialize a Component object (Peak or Background).

    Parameters
    ----------
    component : Component
        Component object to serialize.
    metadata : Metadata, optional
        Metadata associated with the component.

    Returns
    -------
    dict[str, Any]
        Serialized component dictionary.
    """
    # Determine component type
    if isinstance(component, Peak):
        obj_type = "Peak"
    elif isinstance(component, Background):
        obj_type = "Background"
    else:
        obj_type = "Component"

    # Serialize parameters
    parameters: dict[str, dict[str, Any]] = {}
    for param_name, param in component.parameters.items():
        param_dict: dict[str, Any] = {
            "value": param.value,
            "lower": _inf_cast(param.lower),
            "upper": _inf_cast(param.upper),
            "vary": param.vary,
            "expr": param.expr,
        }
        parameters[param_name] = param_dict

    result: dict[str, Any] = {
        "id": component.id_,
        "parent_id": component.parent_id,
        "type": obj_type,
        "model_name": component.model.name,
        "parameters": parameters,
    }
    if component.name is not None:
        result["name"] = component.name

    # Add metadata if available
    if metadata is not None:
        result["metadata"] = asdict(metadata)
    else:
        result["metadata"] = None

    return result


def _inf_cast(value: float) -> str | float:
    """Cast a float to a string representation of infinity."""
    if value == np.inf:
        return "inf"
    if value == -np.inf:
        return "-inf"
    return value


def _json_default(obj: Any) -> Any:
    """
    JSON encoder default handler for non-serializable types.

    Parameters
    ----------
    obj : Any
        Object to encode.

    Returns
    -------
    Any
        JSON-serializable representation.
    """
    if isinstance(obj, (np.integer, np.floating)):
        if np.isinf(obj):
            if obj > 0:
                return "inf"
            return "-inf"
        if np.isnan(obj):
            return None
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def serialize(
    collection: CoreCollection, metadata_service: MetadataService | None = None
) -> dict[str, Any]:
    """
    Serialize a CoreCollection to a dictionary.

    Parameters
    ----------
    collection : CoreCollection
        Collection to serialize.
    metadata_service : MetadataService, optional
        Service providing metadata for objects. If None, no metadata
        will be included.

    Returns
    -------
    dict[str, Any]
        Serialized collection dictionary with version and objects list.
    """
    objects: list[dict[str, Any]] = []

    for obj in collection.objects_index.values():
        # Get metadata if service is available
        metadata: Metadata | None = None
        if metadata_service is not None:
            metadata = metadata_service.get_metadata(obj.id_)

        # Dispatch to appropriate serializer
        if isinstance(obj, Spectrum):
            serialized = _serialize_spectrum(obj, metadata)
        elif isinstance(obj, Region):
            serialized = _serialize_region(obj, metadata)
        elif isinstance(obj, Component):
            serialized = _serialize_component(obj, metadata)
        else:
            # Fallback for unknown types
            serialized = {
                "id": obj.id_,
                "parent_id": getattr(obj, "parent_id", None),
                "type": type(obj).__name__,
                "metadata": asdict(metadata) if metadata else None,
            }

        objects.append(serialized)

    return {
        "version": VERSION,
        "objects": objects,
        "post_processing": [
            serialize_series(series) for series in collection.post_processing.values()
        ],
    }


def _resolve_load_use_gzip(path: Path, use_gzip: bool | None) -> bool:
    """
    Decide whether to read the file as gzip-compressed JSON.

    Parameters
    ----------
    path : Path
        File path.
    use_gzip : bool or None
        If True, read as gzip. If False, read as plain text. If None, detect
        from the ``.gz`` suffix or gzip magic bytes (0x1f 0x8b).

    Returns
    -------
    bool
        Whether to open with gzip.
    """
    if use_gzip is True:
        return True
    if use_gzip is False:
        return False
    if path.suffix.lower() == ".gz":
        return True
    with path.open("rb") as f:
        magic = f.read(2)
    return magic == b"\x1f\x8b"


def dump(
    collection: CoreCollection,
    fp: str | Path,
    metadata_service: MetadataService | None = None,
    indent: int | None = None,
    *,
    use_gzip: bool = False,
    compresslevel: int = 9,
) -> None:
    """
    Serialize a CoreCollection to JSON and save it to a file.

    Writes to a temporary file in the same directory, then replaces the
    destination atomically via :func:`os.replace` so a crash or I/O error
    mid-write cannot leave a truncated document.

    Parameters
    ----------
    collection : CoreCollection
        Collection to serialize.
    fp : str or Path
        Path to save JSON file.
    metadata_service : MetadataService, optional
        Service providing metadata for objects.
    indent : int or None, optional
        JSON indentation level.
    use_gzip : bool, optional
        If True, write gzip-compressed JSON (default False).
    compresslevel : int, optional
        Gzip compression level (0--9), used when ``use_gzip`` is True.
    """
    path = Path(fp)
    data = serialize(collection, metadata_service)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        os.close(fd)
        if use_gzip:
            with gzip.open(tmp_name, "wt", encoding="utf-8", compresslevel=compresslevel) as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=_json_default)
        else:
            with Path(tmp_name).open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=_json_default)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _convert_json_value(value: Any) -> Any:
    r"""
    Convert JSON values back to numpy-compatible types.

    Parameters
    ----------
    value : Any
        JSON value to convert.

    Returns
    -------
    Any
        Converted value (handles \"inf\", \"-inf\" strings and None).
    """
    if isinstance(value, str):
        if value.lower() == "inf":
            return np.inf
        if value.lower() == "-inf":
            return -np.inf
    return value


def _deserialize_spectrum(obj_data: dict[str, Any]) -> Spectrum:
    """
    Deserialize a Spectrum object from dictionary.

    Parameters
    ----------
    obj_data : dict[str, Any]
        Serialized spectrum data.

    Returns
    -------
    Spectrum
        Reconstructed Spectrum object.
    """
    # Reconstruct x-axis from start, step, num_points
    x_start = obj_data["x_start"]
    x_step = obj_data["x_step"]
    x_num_points = obj_data["x_num_points"]

    if "x" in obj_data:
        x = np.asarray(obj_data["x"], dtype=np.float64)
        if x.ndim != 1 or len(x) != x_num_points or not np.all(np.isfinite(x)):
            raise ValueError("Invalid explicit spectrum energy axis")
    elif x_num_points == 0:
        x = np.array([], dtype=np.float64)
    elif x_num_points == 1:
        x = np.array([x_start], dtype=np.float64)
    else:
        x = np.linspace(x_start, x_start + x_step * (x_num_points - 1), x_num_points)

    # Reconstruct y array
    y = np.array(obj_data["y"], dtype=np.float64)
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("Spectrum energy and intensity arrays must have equal lengths")

    # Create Spectrum with explicit ID
    spectrum = Spectrum(x=x, y=y, id_=obj_data["id"])

    return spectrum


def _deserialize_region(obj_data: dict[str, Any]) -> Region:
    """
    Deserialize a Region object from dictionary.

    Parameters
    ----------
    obj_data : dict[str, Any]
        Serialized region data.

    Returns
    -------
    Region
        Reconstructed Region object.
    """
    slice_start = obj_data["slice_start"]
    slice_stop = obj_data["slice_stop"]
    region_slice = slice(slice_start, slice_stop)

    region = Region(slice_=region_slice, parent_id=obj_data["parent_id"], id_=obj_data["id"])

    return region


def _deserialize_component(obj_data: dict[str, Any]) -> Peak | Background:
    """
    Deserialize a Component object (Peak or Background) from dictionary.

    Parameters
    ----------
    obj_data : dict[str, Any]
        Serialized component data.

    Returns
    -------
    Peak or Background
        Reconstructed component object.

    Raises
    ------
    KeyError
        If model name is not found in ModelRegistry.
    """
    model_name = obj_data["model_name"]
    model = ModelRegistry.get(model_name)

    # Extract parameter values from serialized parameters
    param_values: dict[str, float] = {}
    for param_name, param_data in obj_data["parameters"].items():
        param_values[param_name] = float(param_data["value"])

    # Determine component type and create
    obj_type = obj_data["type"]
    display_name = obj_data.get("name")
    if obj_type == "Peak":
        if not isinstance(model, BasePeakModel):
            raise TypeError(f"Model {model_name!r} is not a Peak model")
        component = Peak(
            model=model,
            region_id=obj_data["parent_id"],
            component_id=obj_data["id"],
            name=display_name,
            **param_values,
        )
    elif obj_type == "Background":
        if not isinstance(model, BaseBackgroundModel):
            raise TypeError(f"Model {model_name!r} is not a Background model")
        component = Background(
            model=model,
            region_id=obj_data["parent_id"],
            component_id=obj_data["id"],
            name=display_name,
            **param_values,
        )
    else:
        raise ValueError(f"Unknown component type: {obj_type}")

    # Restore parameter bounds and other attributes
    for param_name, param_data in obj_data["parameters"].items():
        param = component.get_param(param_name)
        lower = _convert_json_value(param_data.get("lower", -np.inf))
        upper = _convert_json_value(param_data.get("upper", np.inf))
        vary = param_data.get("vary", True)
        expr = param_data.get("expr", None)

        param.set(lower=lower, upper=upper, vary=vary, expr=expr)

    return component


def _create_metadata(obj_data: dict[str, Any], obj_type: str) -> Metadata | None:
    r"""
    Create metadata object from serialized data.

    Parameters
    ----------
    obj_data : dict[str, Any]
        Serialized object data.
    obj_type : str
        Type of object (\"Spectrum\", \"Region\", \"Peak\", \"Background\").

    Returns
    -------
    Metadata or None
        Reconstructed metadata object or None if no metadata.
    """
    metadata_dict = obj_data.get("metadata")
    if metadata_dict is None:
        return None

    if obj_type == "Spectrum":
        return SpectrumMetadata(**metadata_dict)
    if obj_type == "Region":
        if metadata_dict:
            return RegionMetadata()
        return None
    if obj_type == "Peak":
        return PeakMetadata(**metadata_dict)
    if obj_type == "Background":
        if metadata_dict:
            return BackgroundMetadata()
        return None

    return None


def _deserialize_objects(
    data: dict[str, Any], collection: CoreCollection, metadata_service: MetadataService
) -> None:
    """Populate staging stores, preserving existing object IDs on append."""
    objects_data = data.get("objects", [])
    if not isinstance(objects_data, list):
        raise ValueError("Document objects must be a list")
    type_order = {"Spectrum": 0, "Region": 1, "Peak": 2, "Background": 2, "Component": 2}
    sorted_objects = sorted(objects_data, key=lambda obj: type_order.get(obj.get("type", ""), 99))
    for obj_data in sorted_objects:
        obj_type = obj_data.get("type")
        obj_id = obj_data.get("id")
        if obj_id is None:
            raise ValueError("Object missing 'id' field")
        if obj_id in collection.objects_index:
            continue
        if obj_type == "Spectrum":
            obj = _deserialize_spectrum(obj_data)
        elif obj_type == "Region":
            parent_id = obj_data.get("parent_id")
            if parent_id and parent_id not in collection.objects_index:
                raise ValueError(f"Region {obj_id} references non-existent parent {parent_id}")
            obj = _deserialize_region(obj_data)
        elif obj_type in ("Peak", "Background", "Component"):
            parent_id = obj_data.get("parent_id")
            if parent_id and parent_id not in collection.objects_index:
                raise ValueError(f"Component {obj_id} references non-existent parent {parent_id}")
            obj = _deserialize_component(obj_data)
        else:
            raise ValueError(f"Unknown object type: {obj_type}")
        collection.add(obj)
        metadata = _create_metadata(obj_data, obj_type)
        if metadata is not None:
            metadata_service.set_metadata(obj_id, metadata)


def deserialize(
    data: dict[str, Any],
    collection: CoreCollection | None = None,
    metadata_service: MetadataService | None = None,
    *,
    mode: LoadMode = "replace",
) -> DeserializeResult:
    """Load a complete document transactionally, accepting versions 1.0 and 1.1.

    New returns a fresh (collection, metadata) pair. Replace requires both
    existing stores and updates them in place. Append preserves existing object
    IDs and gives incoming series fresh IDs on collisions; their snapshot IDs
    follow the new series ID. Failed validation leaves the target untouched.
    Missing source objects and stale fingerprints are retained for inspection.
    """
    if mode not in ("append", "replace", "new"):
        raise ValueError(f"Invalid load mode: {mode}")
    if data.get("version") not in ("1.0", VERSION):
        raise ValueError(f"Version mismatch: expected 1.0 or {VERSION}, got {data.get('version')}")
    if mode == "new" and (collection is not None or metadata_service is not None):
        raise ValueError("mode='new' requires collection and metadata_service to be None")
    if mode == "replace" and (collection is None or metadata_service is None):
        raise ValueError("mode='replace' requires both collection and metadata_service")
    if data["version"] == "1.0" and data.get("post_processing"):
        raise ValueError("Version 1.0 cannot contain post-processing series")
    incoming = deserialize_section(data.get("post_processing", []))
    staged = CoreCollection()
    staged_metadata = MetadataService(staged)
    if mode == "append" and collection is not None:
        staged.objects_index.update(collection.objects_index)
        staged.post_processing.update(collection.post_processing)
        if metadata_service is not None:
            for obj_id in collection.objects_index:
                metadata = metadata_service.get_metadata(obj_id)
                if metadata is not None:
                    staged_metadata.set_metadata(obj_id, metadata)
    _deserialize_objects(data, staged, staged_metadata)
    for series in incoming:
        if series.id_ in staged.post_processing:
            series_id = uuid4().hex
            while series_id in staged.post_processing:
                series_id = uuid4().hex
            result = series.result
            if result is not None:
                result = replace(result, series=replace(result.series, id_=series_id))
            series = replace(series, id_=series_id, result=result)
        staged.post_processing[series.id_] = series
    if mode == "new":
        return staged, staged_metadata
    if collection is None:
        collection = CoreCollection()
    collection.clear()
    collection.objects_index.update(staged.objects_index)
    collection.post_processing.update(staged.post_processing)
    if metadata_service is not None:
        metadata_service.clear()
        for obj_id in staged.objects_index:
            metadata = staged_metadata.get_metadata(obj_id)
            if metadata is not None:
                metadata_service.set_metadata(obj_id, metadata)
    return collection


def load(
    fp: str | Path,
    collection: CoreCollection | None = None,
    metadata_service: MetadataService | None = None,
    *,
    mode: LoadMode = "replace",
    use_gzip: bool | None = None,
) -> DeserializeResult:
    r"""
    Deserialize a JSON file to a CoreCollection (and optionally metadata).

    Parameters
    ----------
    fp : str or Path
        Path to JSON file (plain or gzip-compressed).
    collection : CoreCollection, optional
        Existing collection. Semantics as in deserialize (see mode).
    metadata_service : MetadataService, optional
        Service for storing metadata. Semantics as in deserialize (see mode).
    mode : {\"append\", \"replace\", \"new\"}, optional
        Same as deserialize: append, replace (default), or new.
    use_gzip : bool or None, optional
        If True, read as gzip. If False, read as plain text. If None (default),
        detect from the ``.gz`` suffix or gzip magic bytes.

    Returns
    -------
    CoreCollection or tuple[CoreCollection, MetadataService]
        For append/replace: the collection. For new: (collection, metadata_service).
    """
    path = Path(fp)
    gz = _resolve_load_use_gzip(path, use_gzip)
    if gz:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    return deserialize(data, collection, metadata_service, mode=mode)

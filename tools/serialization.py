"""
Serialization tools for CoreCollection objects.

Provides functionality to serialize CoreCollection instances to JSON format,
preserving object relationships, metadata, and data in a compact representation.
"""

import json
from dataclasses import asdict
from typing import Any, Literal, Optional, Union
from pathlib import Path

import numpy as np

from core.collection import CoreCollection
from core.objects import Spectrum, Region, Peak, Background, Component
from core.services import MetadataService
from core.metadata import Metadata, SpectrumMetadata, RegionMetadata, PeakMetadata, BackgroundMetadata
from core.math_models import ModelRegistry

LoadMode = Literal["append", "replace", "new"]
DeserializeResult = Union[CoreCollection, tuple[CoreCollection, MetadataService]]


class CollectionSerializer:
    """
    Serializer for CoreCollection objects to JSON format.

    Serializes all objects in a collection including their relationships,
    metadata, and data. Optimizes spectrum x-axis storage by using
    start, step, and num_points instead of full arrays.

    Attributes
    ----------
    VERSION : str
        Serialization format version.
    """

    VERSION = "1.0"

    def _serialize_spectrum(self, spectrum: Spectrum, metadata: Optional[Metadata] = None) -> dict[str, Any]:
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
            x_step = float(x[1] - x[0])
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

        # Add metadata if available
        if metadata is not None:
            result["metadata"] = asdict(metadata)
        else:
            result["metadata"] = None

        return result

    def _serialize_region(self, region: Region, metadata: Optional[Metadata] = None) -> dict[str, Any]:
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

    def _serialize_component(
        self, component: Component, metadata: Optional[Metadata] = None
    ) -> dict[str, Any]:
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
                "lower": param.lower,
                "upper": param.upper,
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

        # Add metadata if available
        if metadata is not None:
            result["metadata"] = asdict(metadata)
        else:
            result["metadata"] = None

        return result

    @staticmethod
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
                else:
                    return "-inf"
            elif np.isnan(obj):
                return None
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def serialize(
        self,
        collection: CoreCollection,
        metadata_service: Optional[MetadataService] = None,
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
            metadata: Optional[Metadata] = None
            if metadata_service is not None:
                metadata = metadata_service.get_metadata(obj.id_)

            # Dispatch to appropriate serializer
            if isinstance(obj, Spectrum):
                serialized = self._serialize_spectrum(obj, metadata)
            elif isinstance(obj, Region):
                serialized = self._serialize_region(obj, metadata)
            elif isinstance(obj, Component):
                serialized = self._serialize_component(obj, metadata)
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
            "version": self.VERSION,
            "objects": objects,
        }

    def dump(
        self,
        collection: CoreCollection,
        fp: str | Path,
        metadata_service: Optional[MetadataService] = None,
        indent: int | None = None,
    ):
        """
        Serialize a CoreCollection to JSON string and optionally save to file.

        Parameters
        ----------
        collection : CoreCollection
            Collection to serialize.
        filepath : str | Path
            Path to save JSON file.
        metadata_service : MetadataService, optional
            Service providing metadata for objects.
        indent : int | str | None , default=None
            JSON indentation level.
        """
        fp = Path(fp)
        data = self.serialize(collection, metadata_service)
        fp.parent.mkdir(parents=True, exist_ok=True)
        with fp.open("w") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=self._json_default)

    def _convert_json_value(self, value: Any) -> Any:
        """
        Convert JSON values back to numpy-compatible types.

        Parameters
        ----------
        value : Any
            JSON value to convert.

        Returns
        -------
        Any
            Converted value (handles "inf", "-inf" strings and None).
        """
        if isinstance(value, str):
            if value.lower() == "inf":
                return np.inf
            elif value.lower() == "-inf":
                return -np.inf
        return value

    def _deserialize_spectrum(self, obj_data: dict[str, Any]) -> Spectrum:
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

        if x_num_points == 0:
            x = np.array([], dtype=np.float64)
        elif x_num_points == 1:
            x = np.array([x_start], dtype=np.float64)
        else:
            x = np.linspace(x_start, x_start + x_step * (x_num_points - 1), x_num_points)

        # Reconstruct y array
        y = np.array(obj_data["y"], dtype=np.float64)

        # Create Spectrum with explicit ID
        spectrum = Spectrum(x=x, y=y, id_=obj_data["id"])

        return spectrum

    def _deserialize_region(self, obj_data: dict[str, Any]) -> Region:
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

    def _deserialize_component(self, obj_data: dict[str, Any]) -> Peak | Background:
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
        if obj_type == "Peak":
            component = Peak(
                model=model, region_id=obj_data["parent_id"], component_id=obj_data["id"], **param_values
            )
        elif obj_type == "Background":
            component = Background(
                model=model, region_id=obj_data["parent_id"], component_id=obj_data["id"], **param_values
            )
        else:
            raise ValueError(f"Unknown component type: {obj_type}")

        # Restore parameter bounds and other attributes
        for param_name, param_data in obj_data["parameters"].items():
            param = component.get_param(param_name)
            lower = self._convert_json_value(param_data.get("lower", -np.inf))
            upper = self._convert_json_value(param_data.get("upper", np.inf))
            vary = param_data.get("vary", True)
            expr = param_data.get("expr", None)

            param.set(lower=lower, upper=upper, vary=vary, expr=expr)

        return component

    def _create_metadata(self, obj_data: dict[str, Any], obj_type: str) -> Optional[Metadata]:
        """
        Create metadata object from serialized data.

        Parameters
        ----------
        obj_data : dict[str, Any]
            Serialized object data.
        obj_type : str
            Type of object ("Spectrum", "Region", "Peak", "Background").

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
        elif obj_type == "Region":
            if metadata_dict:  # Check if dict is not empty
                return RegionMetadata()
            return None
        elif obj_type == "Peak":
            return PeakMetadata(**metadata_dict)
        elif obj_type == "Background":
            if metadata_dict:  # Check if dict is not empty
                return BackgroundMetadata()
            return None

        return None

    def deserialize(
        self,
        data: dict[str, Any],
        collection: Optional[CoreCollection] = None,
        metadata_service: Optional[MetadataService] = None,
        *,
        mode: LoadMode = "replace",
    ) -> DeserializeResult:
        """
        Deserialize a dictionary to a CoreCollection (and optionally metadata).

        Parameters
        ----------
        data : dict[str, Any]
            Serialized collection dictionary.
        collection : CoreCollection, optional
            Existing collection. Required for replace; used for append; must be
            None for new.
        metadata_service : MetadataService, optional
            Service for storing metadata. Required for replace; used for append;
            must be None for new.
        mode : {"append", "replace", "new"}, optional
            - "append": Add deserialized objects to existing collection/metadata;
              skip objects that already exist by id. collection optional (new
              if None); metadata_service optional.
            - "replace": Clear collection and metadata in-place, then deserialize
              into the same instances. Requires both collection and metadata_service.
            - "new": Create new CoreCollection and MetadataService, deserialize
              into them. Requires collection is None and metadata_service is None.
              Returns (collection, metadata_service).

        Returns
        -------
        CoreCollection or tuple[CoreCollection, MetadataService]
            For append/replace: the collection. For new: (collection, metadata_service).

        Raises
        ------
        ValueError
            If version is incompatible, data format is invalid, or mode
            requirements are not met (e.g. new with collection provided).
        """
        if mode not in ("append", "replace", "new"):
            raise ValueError(f"mode must be 'append', 'replace', or 'new'; got {mode!r}")

        version = data.get("version", "unknown")
        if version != self.VERSION:
            raise ValueError(
                f"Version mismatch: expected {self.VERSION}, got {version}. "
                "Deserialization may not work correctly."
            )

        if mode == "new":
            if collection is not None or metadata_service is not None:
                raise ValueError("mode='new' requires collection and metadata_service to be None")
            collection = CoreCollection()
            metadata_service = MetadataService(collection)
        elif mode == "replace":
            if collection is None or metadata_service is None:
                raise ValueError("mode='replace' requires both collection and metadata_service")
            collection.clear()
            metadata_service.clear()

        # For append: collection may be None (create new); metadata_service optional
        if mode == "append" and collection is None:
            collection = CoreCollection()

        objects_data = data.get("objects", [])
        if not objects_data:
            if mode == "new":
                return (collection, metadata_service)
            return collection

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
                obj = self._deserialize_spectrum(obj_data)
            elif obj_type == "Region":
                parent_id = obj_data.get("parent_id")
                if parent_id and parent_id not in collection.objects_index:
                    raise ValueError(f"Region {obj_id} references non-existent parent {parent_id}")
                obj = self._deserialize_region(obj_data)
            elif obj_type in ("Peak", "Background", "Component"):
                parent_id = obj_data.get("parent_id")
                if parent_id and parent_id not in collection.objects_index:
                    raise ValueError(f"Component {obj_id} references non-existent parent {parent_id}")
                obj = self._deserialize_component(obj_data)
            else:
                raise ValueError(f"Unknown object type: {obj_type}")

            collection.add(obj)

            if metadata_service is not None:
                metadata = self._create_metadata(obj_data, obj_type)
                if metadata is not None:
                    metadata_service.set_metadata(obj_id, metadata)

        if mode == "new":
            return (collection, metadata_service)
        return collection

    def load(
        self,
        fp: str | Path,
        collection: Optional[CoreCollection] = None,
        metadata_service: Optional[MetadataService] = None,
        *,
        mode: LoadMode = "replace",
    ) -> DeserializeResult:
        """
        Deserialize a JSON file to a CoreCollection (and optionally metadata).

        Parameters
        ----------
        fp : str or Path
            Path to JSON file.
        collection : CoreCollection, optional
            Existing collection. Semantics as in deserialize (see mode).
        metadata_service : MetadataService, optional
            Service for storing metadata. Semantics as in deserialize (see mode).
        mode : {"append", "replace", "new"}, optional
            Same as deserialize: append, replace (default), or new.

        Returns
        -------
        CoreCollection or tuple[CoreCollection, MetadataService]
            For append/replace: the collection. For new: (collection, metadata_service).
        """
        fp = Path(fp)
        with fp.open("r") as f:
            data = json.load(f)
        return self.deserialize(data, collection, metadata_service, mode=mode)

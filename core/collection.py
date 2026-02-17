from typing import Dict, TypeVar

from .objects import CoreObject, Spectrum, Region

T = TypeVar("T")


class CoreCollection:
    """
    Collection and lifecycle manager for core objects.

    `CoreCollection` is a lightweight registry that stores and indexes
    all core objects participating in the model:
    - Spectrum
    - Region
    - Peak
    - Background

    The collection is responsible for:
    - Object registration and lookup by ID
    - Maintaining parent–child relationships via `parent_id`
    - Cascading removal of hierarchical objects

    It is intentionally *not* responsible for:
    - Validation of scientific correctness
    - Business logic (copy/paste, fitting, optimization)
    - Object creation policies

    Design principles
    -----------------
    - Single source of truth: all objects are stored in a single index.
    - Relationships are implicit and resolved via `parent_id`.
    - No reliance on ID prefixes or explicit type dispatch.
    - Minimal assumptions about object internals.

    Object hierarchy assumptions
    -----------------------------
    The collection assumes the following containment hierarchy:

        Spectrum
          └── Region
                └── Peak / Background / other components

    These rules are not enforced structurally, but are relied upon
    for correct cascading deletion.

    Notes
    -----
    - All objects must expose:
        - `id_ : str`
        - `parent_id : Optional[str]`
    - IDs must be globally unique within the collection.
    - Objects are assumed to be immutable with respect to identity
      (`id_` and `parent_id` should not change after registration).

    Attributes
    ----------
    objects_index : dict[str, Spectrum | Region | Peak]
        Global mapping from object ID to core object instance.
    """

    def __init__(self):
        self.objects_index: Dict[str, CoreObject] = {}

    def add(self, obj: CoreObject) -> None:
        """
        Register an object in the collection.

        This method adds the object to the internal index.
        It does NOT automatically add parents or children.

        Parameters
        ----------
        obj : Spectrum or Region or Peak or Background
            core object to register.

        Raises
        ------
        ValueError
            If the object does not have a valid `id_`.
        KeyError
            If an object with the same ID already exists.
        """
        if obj.id_ is None:
            raise ValueError("Object must have an id_")

        if obj.id_ in self.objects_index:
            raise KeyError(f"Object with id '{obj.id_}' already exists")

        self.objects_index[obj.id_] = obj

    def remove(self, obj: CoreObject | str) -> list[CoreObject]:
        """
        Remove an object from the collection and return the removed objects.

        Removal is recursive:
        - If a Spectrum is removed, all its Regions and Peaks are removed.
        - If a Region is removed, all its Peaks (and Backgrounds) are removed.
        - If a Peak or Background is removed, only the object itself is removed.

        Parameters
        ----------
        obj : Spectrum or Region or Peak or Background or str
            Object instance or its ID.

        Returns
        -------
        list[CoreObject]
            The removed objects.

        Raises
        ------
        KeyError
            If the object ID is not present in the collection.
        """
        if isinstance(obj, str):
            obj_id = obj
        else:
            obj_id = obj.id_

        obj = self.objects_index.pop(obj_id)
        removed_objects: list[CoreObject] = [obj]

        if isinstance(obj, (Spectrum, Region)):
            children = list(self.get_children(obj_id))
            for ch in children:
                removed_objects.extend(self.remove(ch))

        return removed_objects

    def get(self, obj_id: str) -> CoreObject:
        """
        Retrieve an object by its ID.

        Parameters
        ----------
        obj_id : str
            Identifier of the requested object.

        Returns
        -------
        Spectrum or Region or Peak or Background
            The corresponding core object.

        Raises
        ------
        KeyError
            If no object with this ID exists in the collection.
        """
        return self.objects_index[obj_id]

    def get_typed(self, obj_id: str, tp: type[T]) -> T:
        """
        Retrieve an object by its ID and check type.

        Parameters
        ----------
        obj_id : str
            Identifier of the requested object.
        tp : type
            Requested object type.
            Should be Spectrum or Region or Peak or Background

        Returns
        -------
        obj : Spectrum | Region | Peak | Background
            The corresponding core object.

        Raises
        ------
        TypeError
            If requested object and requested type dont match.
        """
        obj = self.get(obj_id)
        if not isinstance(obj, tp):
            raise TypeError(f"{obj_id} is not {tp.__name__}")
        return obj

    def get_children(self, obj_id: str) -> tuple[CoreObject, ...]:
        """
        Retrieve direct children of a given object.

        Children are defined as objects whose `parent_id`
        matches the given `obj_id`.

        Parameters
        ----------
        obj_id : str
            Parent object ID.

        Returns
        -------
        tuple of Region or Peak or Background
            Direct children of the given object.
            The order is not guaranteed.
        """
        return tuple(ch for ch in self.objects_index.values() if ch.parent_id == obj_id)

    def get_subtree(self, obj_id: str) -> tuple[CoreObject, ...]:
        """
        Return the object and all descendants without removing.

        Same set of objects that would be returned by remove(obj_id).

        Parameters
        ----------
        obj_id : str
            Root object ID.

        Returns
        -------
        tuple[CoreObject, ...]
            The object and all its descendants.
        """
        obj = self.objects_index[obj_id]
        result: list[CoreObject] = [obj]
        for ch in self.get_children(obj_id):
            result.extend(self.get_subtree(ch.id_))
        return tuple(result)

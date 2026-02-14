from dataclasses import dataclass
from core.services import (
    CollectionQueryService,
    SpectrumService,
    RegionService,
    DataQueryService,
    ComponentService,
    MetadataService,
)
from core.collection import CoreCollection


@dataclass(frozen=True)
class ApplicationContext:
    """Read and write access to core services for command execution."""

    collection: CollectionQueryService
    spectrum: SpectrumService
    region: RegionService
    data: DataQueryService
    component: ComponentService
    metadata: MetadataService

    @classmethod
    def from_collection(cls, collection: CoreCollection) -> "ApplicationContext":
        """Build context from a core collection."""
        return cls(
            collection=CollectionQueryService(collection),
            spectrum=SpectrumService(collection),
            region=RegionService(collection),
            data=DataQueryService(collection),
            component=ComponentService(collection),
            metadata=MetadataService(collection),
        )

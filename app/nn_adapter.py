"""
NN pipeline adapters: convert pipeline outputs to Change objects.

SegmenterPipelineAdapter converts SegmenterResult list to CreateRegion,
CreateBackground, and CreatePeak changes for CommandExecutor.
"""

from uuid import uuid4

from tools.nn.segmenter import SegmenterResult

from .command.changes import (
    CompositeChange,
    CreateBackground,
    CreatePeak,
    CreateRegion,
)


def segmenter_results_to_changes(
    spectrum_id: str,
    results: list[SegmenterResult],
) -> CompositeChange:
    """
    Convert segmenter pipeline results to a CompositeChange.

    For each SegmenterResult, generates a region_id first, then creates
    CreateRegion, CreateBackground (if present), and CreatePeak changes.
    Order ensures region exists before children are created.

    Parameters
    ----------
    spectrum_id : str
        Identifier of the parent spectrum.
    results : list[SegmenterResult]
        Segmenter pipeline output.

    Returns
    -------
    CompositeChange
        Changes for CommandExecutor: CreateRegion, CreateBackground, CreatePeak.
    """
    changes: list[CreateRegion | CreateBackground | CreatePeak] = []

    for result in results:
        region_id = f"r{uuid4().hex}"

        changes.append(
            CreateRegion(
                spectrum_id=spectrum_id,
                start=result.region.start,
                stop=result.region.stop,
                region_id=region_id,
            )
        )

        if result.background is not None:
            changes.append(
                CreateBackground(
                    region_id=region_id,
                    model_name=result.background.model_name,
                    parameters=result.background.parameters,
                )
            )

        for peak in result.peaks:
            changes.append(
                CreatePeak(
                    region_id=region_id,
                    model_name=peak.model_name,
                    parameters=peak.parameters,
                )
            )

    return CompositeChange(changes=changes)

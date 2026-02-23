import pytest
import numpy as np

from tools.dto import DTOService, ComponentDTO, RegionDTO, SpectrumDTO, ParameterDTO


@pytest.fixture
def srv(simple_collection):
    return DTOService(simple_collection)


def test_get_component_returns_component_dto(srv, region_id, peak_id, background_id):
    dto = srv.get_component(peak_id)

    assert isinstance(dto, ComponentDTO)
    assert dto.id_ == peak_id
    assert dto.parent_id == region_id
    assert dto.normalized is False
    assert dto.kind == "peak"

    dto = srv.get_component(background_id)
    assert isinstance(dto, ComponentDTO)
    assert dto.id_ == background_id
    assert dto.parent_id == region_id
    assert dto.normalized is False
    assert dto.kind == "background"


def test_get_component_parameters_are_parameter_dtos(srv, peak_id):
    dto = srv.get_component(peak_id)

    assert isinstance(dto.parameters, dict)
    assert all(isinstance(p, ParameterDTO) for p in dto.parameters.values())


def test_get_component_is_immutable(srv, peak_id):
    dto = srv.get_component(peak_id)

    with pytest.raises(Exception):
        dto.parameters["cen"].value = 10.0


def test_get_component_normalized_flag(srv, peak_id):
    dto = srv.get_component(peak_id, normalize=True)

    assert dto.normalized is True


def test_normalized_component_differs_from_raw(srv, peak_id):
    raw = srv.get_component(peak_id, normalize=False)
    norm = srv.get_component(peak_id, normalize=True)

    assert raw.parameters["amp"].value != norm.parameters["amp"].value


def test_get_region_returns_region_dto(srv, region_id):
    dto = srv.get_region(region_id)

    assert isinstance(dto, RegionDTO)
    assert dto.id_ == region_id
    assert dto.normalized is False


def test_get_region_data_matches_slice(srv, region_id):
    dto = srv.get_region(region_id)

    assert isinstance(dto.x, np.ndarray)
    assert isinstance(dto.y, np.ndarray)
    assert len(dto.x) == len(dto.y)


def test_get_region_repr_returns_region_and_components(srv, region_id):
    reg_dto, components = srv.get_region_repr(region_id)

    assert isinstance(reg_dto, RegionDTO)
    assert isinstance(components, tuple)
    assert len(components) == 2
    assert all(isinstance(c, ComponentDTO) for c in components)


def test_get_spectrum_returns_spectrum_dto(srv, spectrum_id):
    dto = srv.get_spectrum(spectrum_id)

    assert isinstance(dto, SpectrumDTO)
    assert dto.id_ == spectrum_id
    assert dto.parent_id is None


def test_region_spectrum_array_immutable(srv, region_id, spectrum_id):
    dto = srv.get_region(region_id)
    with pytest.raises(Exception):
        x, y = dto.x, dto.y
        y += 1

    # copies are mutable
    y = y.copy()
    y += 1

    dto = srv.get_spectrum(spectrum_id)
    with pytest.raises(Exception):
        x, y = dto.x, dto.y
        y += 1

    # copies are mutable
    y = y.copy()
    y += 1


def test_get_spectrum_normalized(srv, spectrum_id):
    raw = srv.get_spectrum(spectrum_id, normalize=False)
    norm = srv.get_spectrum(spectrum_id, normalize=True)

    assert not np.allclose(raw.y, norm.y)


def test_get_spectrum_repr_structure(srv, spectrum_id):
    spec_dto, regions = srv.get_spectrum_repr(spectrum_id)

    assert isinstance(spec_dto, SpectrumDTO)
    assert isinstance(regions, tuple)

    for reg_dto, comps in regions:
        assert isinstance(reg_dto, RegionDTO)
        assert isinstance(comps, tuple)

import pytest
import numpy as np
from unittest.mock import MagicMock
from lib.spectra import SpectrumCollection, Spectrum, Region, Peak
from lib.model_tools import SegmenterModelProcessor


@pytest.fixture
def gaussian_data():
    """Создаёт тестовую гауссиану и бинарные маски."""
    x = np.linspace(-5, 5, 256)
    y = np.exp(-(x**2) / 2)  # σ = 1
    sigma = 1.0

    # область ±3σ
    peak_mask = np.abs(x) <= 3 * sigma

    # 3 единицы около максимума (в центре)
    max_mask = np.zeros_like(x, dtype=bool)
    max_center = np.argmax(y)
    max_mask[max_center - 1 : max_center + 2] = True
    return x, y.astype(np.float32), peak_mask, max_mask


@pytest.fixture
def mock_processor(gaussian_data):
    """Создаёт SegmenterModelProcessor c подменённой моделью."""
    collection = SpectrumCollection()
    mock_onnx = MagicMock()
    mock_onnx.run.return_value = [
        np.stack([gaussian_data[2].astype(np.float32), gaussian_data[3].astype(np.float32)])[
            np.newaxis, :, :
        ]
    ]
    # отключаем load_model
    SegmenterModelProcessor.load_model = lambda *x: None
    proc = SegmenterModelProcessor(
        "fake_model.onnx", collection, pred_threshold=0.5, add_mask_smoothing=True
    )
    # подменяем ort_session
    proc.ort_session = mock_onnx
    return proc


# ===========================================================
# predict
# ===========================================================
def test_prepare_input_shapes(mock_processor, gaussian_data):
    x, y, *_ = gaussian_data
    prepared = mock_processor.prepare_input(y)
    assert prepared.shape == (1, 2, len(y))
    assert np.all(prepared >= 0) and np.all(prepared <= 1)


def test_smooth_mask(mock_processor):
    mask = np.zeros(100)
    mask[40:60] = 1
    smoothed = mock_processor.smooth_mask(mask, window_length=10)
    assert smoothed.shape == mask.shape
    assert smoothed.max() <= 1.0


def test_restrict_mask_with_smoothing(mock_processor, gaussian_data):
    x, y, peak_mask, max_mask = gaussian_data
    raw_peak = peak_mask.astype(np.float32)
    raw_max = max_mask.astype(np.float32)
    peakm, maxm = mock_processor.restrict_mask(raw_peak, raw_max)
    assert peakm.dtype == bool and maxm.dtype == bool  # проверка типа
    assert peakm.sum() > 0 and maxm.sum() == 3  # max_mask должен сохранить 3 точки


def test_predict_uses_model_and_threshold(mock_processor, gaussian_data):
    _, y, _, _ = gaussian_data
    peak_mask, max_mask = mock_processor.predict(y)
    # Проверяем, что mock вызывался
    mock_processor.ort_session.run.assert_called_once()
    assert peak_mask.dtype == bool
    assert max_mask.dtype == bool
    assert np.any(peak_mask)
    assert np.any(max_mask)


# ===========================================================
# parse_masks_to_regions
# ===========================================================
def test_find_borders_and_prepare_max_mask(mock_processor):
    mask = np.zeros(50, dtype=bool)
    mask[10:20] = True
    mask[30:35] = True

    borders = mock_processor.find_borders(mask)
    assert np.all(borders >= 0)
    assert len(borders) == 4  # начало и конец двух областей

    med = mock_processor.prepare_max_mask(mask)
    assert np.all((med >= 0) & (med < len(mask)))
    assert len(med) == 2


def test_recalculate_idx(mock_processor):
    x1 = np.linspace(0, 10, 100)
    x2 = np.linspace(0, 10, 200)
    idx = mock_processor.recalculate_idx(50, x1, x2)
    assert 0 <= idx < len(x2)
    # Проверим границу выхода за диапазон
    idx2 = mock_processor.recalculate_idx(999, x1, x2)
    assert idx2 == len(x2) - 1


def test_parse_masks_to_regions_yields_expected_regions(mock_processor, gaussian_data):
    x, y, peak_mask, max_mask = gaussian_data
    # x_int — та же сетка для простоты
    result = list(mock_processor.parse_masks_to_regions(x, x, peak_mask, max_mask))
    assert len(result) == 1  # одна область
    for f, t, maxima in result:
        assert isinstance(f, (int, np.integer))
        assert isinstance(t, (int, np.integer))
        assert np.all((maxima >= 0) & (maxima <= len(x)))


def test_parse_results_creates_region_and_peaks(mock_processor, gaussian_data):
    x, y, peak_mask, max_mask = gaussian_data

    spectrum = Spectrum(x, y)

    results = mock_processor.parse_results(x, x, peak_mask, max_mask, spectrum)
    assert all(isinstance(r, tuple) and isinstance(r[0], Region) for r in results)
    for region, peaks in results:
        assert all(isinstance(p, Peak) for p in peaks)

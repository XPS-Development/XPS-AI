"""
Dialog for CSV export options.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QSpinBox,
    QWidget,
)


if TYPE_CHECKING:
    from .controller import ControllerWrapper


class ExportOptionsDialog(QDialog):
    """
    Dialog that collects CSV export options.
    """

    def __init__(self, *, for_peaks: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Export options")
        layout = QFormLayout(self)

        self._separator = QComboBox(self)
        self._separator.addItem("comma", ",")
        self._separator.addItem("tab", "\t")
        self._separator.addItem("space", " ")
        self._separator.setCurrentText("space")

        self._precision = QSpinBox(self)
        self._precision.setRange(0, 12)
        self._precision.setValue(3)

        layout.addRow("Separator", self._separator)
        layout.addRow("Precision", self._precision)

        if for_peaks:
            self._xps_peak = QCheckBox(self)
            self._xps_peak.setChecked(True)

            layout.addRow("Use XPS peak format", self._xps_peak)
        else:
            self._include_eval = QCheckBox(self)
            self._include_eval.setChecked(False)

            self._include_background = QCheckBox(self)
            self._include_background.setChecked(True)

            self._include_difference = QCheckBox(self)
            self._include_difference.setChecked(True)

            layout.addRow("Include evaluated components", self._include_eval)
            layout.addRow("Include background", self._include_background)
            layout.addRow("Include difference", self._include_difference)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def peak_options(self) -> dict[str, object]:
        """
        Return options for peak export.
        """
        return {
            "separator": str(self._separator.currentData()),
            "precision": int(self._precision.value()),
            "use_xps_peak_names": self._xps_peak.isChecked(),
        }

    def spectrum_options(self) -> dict[str, object]:
        """
        Return options for spectrum export.
        """
        return {
            "separator": str(self._separator.currentData()),
            "precision": int(self._precision.value()),
            "include_evaluated_components": self._include_eval.isChecked(),
            "include_background": self._include_background.isChecked(),
            "include_difference": self._include_difference.isChecked(),
        }


def export_spectra(
    controller: "ControllerWrapper",
    spectra_ids: list[str],
    *,
    parent: QWidget | None = None,
) -> bool:
    """
    Run spectrum export pipeline: options dialog + save target + export.

    Parameters
    ----------
    controller : ControllerWrapper
        Controller used to run exports.
    spectra_ids : list[str]
        Spectra selected for export.
    parent : QWidget or None, optional
        Parent widget for dialogs.

    Returns
    -------
    bool
        True if export command(s) were executed.
    """
    if len(spectra_ids) == 0:
        return False
    options_dialog = ExportOptionsDialog(for_peaks=False, parent=parent)
    if options_dialog.exec() != QDialog.Accepted:
        return False
    options = options_dialog.spectrum_options()
    targets = _select_export_targets(spectra_ids, parent=parent, suffix="")
    if targets is None:
        return False
    for spectrum_id, path in targets:
        controller.export_spectrum(spectrum_id, path, **options)
    return True


def export_peaks(
    controller: "ControllerWrapper",
    spectrum_ids: list[str],
    *,
    parent: QWidget | None = None,
) -> bool:
    """
    Run peak export pipeline: options dialog + save target + export.

    Parameters
    ----------
    controller : ControllerWrapper
        Controller used to run exports.
    spectrum_ids : list[str]
        Spectra selected for export.
    parent : QWidget or None, optional
        Parent widget for dialogs.

    Returns
    -------
    bool
        True if export command(s) were executed.
    """
    if len(spectrum_ids) == 0:
        return False
    options_dialog = ExportOptionsDialog(for_peaks=True, parent=parent)
    if options_dialog.exec() != QDialog.Accepted:
        return False
    options = options_dialog.peak_options()
    targets = _select_export_targets(spectrum_ids, parent=parent, suffix="_peaks")
    if targets is None:
        return False
    for spectrum_id, path in targets:
        controller.export_peak_parameters(spectrum_id, path, **options)
    return True


def _select_export_targets(
    spectrum_ids: list[str],
    *,
    parent: QWidget | None,
    suffix: str,
) -> list[tuple[str, str | Path]] | None:
    """
    Ask for output file/dir depending on number of selected spectra.
    """
    if len(spectrum_ids) == 1:
        title = "Export CSV" if suffix == "" else "Export peak parameters CSV"
        filename, _ = QFileDialog.getSaveFileName(
            parent,
            title,
            "",
            "CSV files (*.csv);;Text files (*.txt);;All files (*)",
        )
        if not filename:
            return None
        return [(spectrum_ids[0], filename)]

    directory = QFileDialog.getExistingDirectory(parent, "Select output directory")
    if not directory:
        return None
    return [
        (spectrum_id, Path(directory) / f"{spectrum_id[:5]}{suffix}.csv") for spectrum_id in spectrum_ids
    ]

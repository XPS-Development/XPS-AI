"""Properties panel with optimize action above the properties tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .assets import icon_path
from .optimize_confirm import confirm_and_optimize
from .properties import PropertiesView

if TYPE_CHECKING:
    from .controller import ControllerWrapper


class PropertiesPanel(QWidget):
    """
    Composite widget: optimize button row + :class:`PropertiesView`.

    Top bar: Optimize regions for the selected spectrum. Per-region optimize
    lives on each region row in the tree.
    """

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        """
        Initialize the properties panel.

        Parameters
        ----------
        controller : ControllerWrapper
            Application controller.
        parent : QWidget or None, optional
            Parent widget.
        """
        super().__init__(parent)
        self.setObjectName("PropertiesPanel")
        self._controller = controller

        self._optimize_regions_btn = QPushButton("Optimize regions", self)
        self._optimize_regions_btn.setIcon(QIcon(str(icon_path("optimize.svg"))))
        self._view = PropertiesView(controller, self)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(4)
        btn_row.addWidget(self._optimize_regions_btn)
        btn_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setSpacing(4)
        layout.addLayout(btn_row)
        layout.addWidget(self._view)

        self.setStyleSheet(
            """
            QWidget#PropertiesPanel {
                background: #f7f7f7;
            }
            QWidget#PropertiesPanel QPushButton {
                background: #ffffff;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 3px 10px;
            }
            QWidget#PropertiesPanel QPushButton:hover {
                background: #f0f0f0;
            }
            QWidget#PropertiesPanel QPushButton:disabled {
                color: #a0a0a0;
                background: #f3f3f3;
            }
            """
        )

        self._optimize_regions_btn.clicked.connect(self._on_optimize_regions)
        self._update_button_state(self._controller.selected_spectrum_id)

    @property
    def view(self) -> PropertiesView:
        """Return the underlying properties tree view."""
        return self._view

    def refresh(self) -> None:
        """Rebuild the properties tree from the controller."""
        self._view.refresh()
        self._update_button_state(self._controller.selected_spectrum_id)

    def on_controller_selection_changed(
        self,
        spectrum_id: str | None,
        region_id: str | None,
        component_id: str | None,
    ) -> None:
        """
        Forward selection changes to the properties view and refresh buttons.

        Parameters
        ----------
        spectrum_id, region_id, component_id : str or None
            Current controller selection.
        """
        self._view.on_controller_selection_changed(spectrum_id, region_id, component_id)
        self._update_button_state(spectrum_id)

    def _update_button_state(self, spectrum_id: object) -> None:
        """Enable Optimize regions when a spectrum with regions is selected."""
        has_spectrum = isinstance(spectrum_id, str) and bool(spectrum_id)
        has_regions = False
        if has_spectrum:
            has_regions = bool(self._controller.query.get_regions_ids(spectrum_id))
        self._optimize_regions_btn.setVisible(has_spectrum)
        self._optimize_regions_btn.setEnabled(has_spectrum and has_regions)

    def _on_optimize_regions(self) -> None:
        """Optimize all regions of the selected spectrum."""
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            return
        confirm_and_optimize(
            self,
            self._controller,
            spectrum_ids=[spectrum_id],
        )

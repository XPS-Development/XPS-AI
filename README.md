# XPS-AI
The XPS-AI project is a comprehensive toolset for analyzing X-ray Photoelectron Spectroscopy (XPS) spectra. The project provides a neural network model for XPS spectra segmentation, data processing and visualization tools for analyzing XPS spectra, and a graphical user interface (GUI) for easy interaction with the tools.

## v0.3.0

This release introduces a fully redesigned application architecture, enabling significant improvements in performance, flexibility, and extensibility.

---

### 🚀 Features

* **Auto fit**: the neural network proposes regions and peaks, then the decomposition is optimized. Initial parameter guesses are up to 10× faster
* Parametric optimization powered by **lmfit**
* Parameter expressions (`expr`): a parameter can be computed from the same parameter of other peaks and backgrounds, for example `2 * pabcd`. A constructor inserts a short unique component id; optimization follows those links and includes the regions they depend on
* Interactive editing: a click on a parameter opens a slider over its soft range and updates the plot while dragging, recorded as one undo step. Region edges can be dragged on the plot, and dedicated modes add a peak or split a region at a click
* Copy decomposition: clone one spectrum's regions, peaks, and backgrounds onto other spectra. Chosen parameters stay linked to the source through expressions; intensities can be rescaled to the target, an existing fit can be overwritten, and an optional optimization can run after the copy
* Flexible export system for analysis results

---

### ⚠️ Limitations

* Post-analysis mode is not yet available in this version
* Supported peak models: Pseudo-Voigt (default), asymmetric Pseudo-Voigt, and tail Pseudo-Voigt
* Supported background models:

  * Linear
  * Constant
  * Shirley

---

### Installation

#### Windows Installer

1. Download `xps-ai_0.3.0_x64.exe`
2. Run the installer
3. Follow the setup instructions

#### Portable Version

1. Extract the archive
2. Run `XPS-AI.exe`

---

### Build from Source

Python 3.11 or newer is required. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

1. Clone the repository:

   ```bash
   git clone https://github.com/XPS-Development/XPS-AI.git
   cd XPS-AI
   ```

2. Install [uv](https://docs.astral.sh/uv/).

3. Install the application and the development tools (tests, lint, type check):

   ```bash
   uv sync --group dev
   ```

4. Run the application:

   ```bash
   uv run python main.py
   ```

For matplotlib-based debugging (`debug/viewer.py`), also sync the interactive group:

```bash
uv sync --group dev --group interactive
```

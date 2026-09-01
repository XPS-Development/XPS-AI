# XPS-AI
The XPS-AI project is a comprehensive toolset for analyzing X-ray Photoelectron Spectroscopy (XPS) spectra. The project provides a neural network model for XPS spectra segmentation, data processing and visualization tools for analyzing XPS spectra, and a graphical user interface (GUI) for easy interaction with the tools.

## v0.1.0 — First Public Release

This release introduces a fully redesigned application architecture, enabling significant improvements in performance, flexibility, and extensibility.

---

### 🚀 Features

* Parametric optimization powered by **lmfit**
* Parameter constraints support
* Fast parameter initialization mechanism for auto-analysis (up to 10× speed improvement)
* Flexible export system for analysis results

---

### ⚠️ Limitations

* Post-analysis mode is not yet available in this version
* Currently supported peak model: Pseudo-Voigt only
* Supported background models:

  * Linear
  * Constant
  * Shirley

---

### Installation

#### Windows Installer

1. Download `xps-ai_0.1.0_x64.exe`
2. Run the installer
3. Follow the setup instructions

#### Portable Version

1. Extract the archive
2. Run `XPS-AI.exe`

---

### Build from Source

1. Clone the repository:

   ```bash
   git clone https://github.com/XPS-Development/XPS-AI.git
   cd XPS-AI
   ```

2. Install [uv](https://docs.astral.sh/uv/).

3. Install dependencies:

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

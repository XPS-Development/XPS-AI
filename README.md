# XPS-AI
The XPS-AI project is a comprehensive toolset for analyzing X-ray Photoelectron Spectroscopy (XPS) spectra. The project provides a neural network model for XPS spectra segmentation, data processing and visualization tools for analyzing XPS spectra, and a graphical user interface (GUI) for easy interaction with the tools.

## Features
* Neural network model for XPS spectra segmentation
* Data processing and visualization tools for analyzing XPS spectra
* Graphical user interface (GUI) for easy interaction with the tools
* Manual analysis support
* Support for exporting spectra data to various formats

## Installation

**Requirements**: Python 3.8

1. Clone git repo:  
`git clone https://github.com/XPS-Development/XPS-AI.git`  
(Optional) Create new python virtual environment:  
`cd XPS-AI`  
`python -m venv venv` (or other methods to create virtual env)

2. Install requirements:  
`pip install -r app_requirements.txt`

3. Run UI:  
`python main.py`  
(Optional) Build app with Nuitka:  
`nuitka --onefile --enable-plugin=pyside6 --include-data-dir=venv\Lib\site-packages\scipy.libs=scipy.libs --include-data-files=model.onnx=model.onnx --windows-console-mode=disable main.py`


## Project Structure
The project structure is as follows:

`model`: Neural network model and related code  
`tools`: Data processing and visualization tools  
`run_train.py`: Script for training the neural network model  
`main.py`: Script for running the GUI application  
`app_requirements.txt`: Requirements file for the GUI application  
`requirements.txt`: Requirements file for the project  
`model.onnx`: deployed model to use with various frameworks  

# CellQt

This repository contains a PyQt5-based desktop platform for several cell image processing tasks.

## Scripts

- `main.py` - Launches the main GUI which aggregates the individual applications under a unified interface with navigation and logging.
- `super_resolution.py` - Provides a GUI to perform single-image super-resolution using PyTorch models. Drag-and-drop images, select scale and model, and reconstruct higher resolution images.
- `microspectra_sr.py` - Implements high spectral microscope image super-resolution with options to load models, process images and visualize results.
- `target_detection.py` - YOLO-based object detection tool for loading custom models and detecting targets in selected images.
- `fluorescence_unmixing.py` - Fluorescence signal unmixing interface that can optionally use the DeepTrans project for model inference.

## Running

Install the dependencies listed in `requirements.txt`. After installing, start the main application with:

```bash
python main.py
```

The optional DeepTrans dependency required for fluorescence unmixing is not available on PyPI. Obtain it by cloning the [DeepTrans-HSU](https://github.com/preetam22n/DeepTrans-HSU) repository and ensure it is accessible on your `PYTHONPATH`.

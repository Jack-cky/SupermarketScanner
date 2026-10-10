# Changelog

## [1.0.6] - 2026-10-10

Added ONNX inference support and streamlined the demo deployment workflow.

### Added
- Added an `OnnxModel` implementation using ONNX Runtime for YOLO segmentation inference.
- Added shared sigmoid, IoU, and non-maximum suppression utilities for model post-processing.
- Added a dedicated `apps/demo/Dockerfile` for building and running the Streamlit demo.
- Added an optional headless OpenCV dependency for container deployments.

### Changed
- Updated the inference engine to load `.onnx`, `.pt`, and `.hef` model formats.
- Moved demo dependencies into the unified `pyproject.toml` configuration.
- Simplified the Makefile to build, publish, run, and stop the demo image directly.
- Updated the README with the latest release date.
- Expanded `.gitignore` coverage for Python tooling, local environments, and project artefacts.

### Removed
- Removed the legacy multi-stage `apps/Dockerfile`.
- Removed Docker Compose orchestration and the separate demo requirements file.

## [1.0.5] - 2026-06-28

Refactored project structure and streamlined the build/deployment process.

### Added
- Added a `kiosk` simulation application in `apps/kiosk/`.
- Added Docker Compose orchestration via `apps/compose.yaml`.
- Added `products.yaml` for centralised product pricing configuration.
- Added more comprehensive `.gitignore` and `.dockerignore` files.

### Changed
- Renamed the core Python package from `supermarketscanner` to `smktscnr`.
- Reorganised the repository architecture by relocating standalone apps to the `apps/` directory.
- Migrated legacy `setup.py` and `requirements.txt` dependencies into a unified `pyproject.toml`.
- Replaced the environment-specific `Makefile` with Docker workflow targets e.g., build, push, demo, kiosk.
- Replaced `03_application.ipynb` with `03_inference.ipynb` and updated previous notebooks.
- Updated apt package dependencies in `packages.txt`.
- Enhanced `README.md` layout and improved descriptions.

### Removed
- Removed deprecated deployment scripts `docker/Dockerfile` and `deploy.sh`.
- Removed outdated `tests/` directory contents.
- Removed duplicated mock images from `demo/static/`.

## [1.0.4] - 2025-06-25

Improved working prototype efficiency and stability.

### Added
- Added a `.dockerignore` file.

### Changed
- Adjusted path settings in the notebook.
- Packaged `supermarketscanner` as a library.
- Updated the `Dockerfile` to minimise the image size.

### Fixed
- Resolved the OpenCV import error in Streamlit Cloud by adding `libgl1-mesa-glx` to the package requirements.

## [1.0.3] - 2025-04-17

Fine-tuned the model and created a demo app.

### Added
- Added test cases for `SupermarketScanner`.
- Created a Streamlit demo app.

### Changed
- Fine-tuned the model with transaction images.
- Restructured the repository.

### Fixed
- Fixed unstable display of transaction summary during video scan.

## [1.0.2] - 2023-07-30

Organised folder structure.

### Removed
- Removed unnecessary files in the repository.

## [1.0.1] - 2023-04-17

Initial repository.

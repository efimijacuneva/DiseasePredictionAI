
# DiseasePredictionAI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

> Disease Prediction using Machine Learning models and a Flask web interface

## Table of Contents
- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset & Credits](#dataset--credits)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the models (pipeline)](#training-the-models-pipeline)
  - [Running the Flask app (inference)](#running-the-flask-app-inference)
- [How it works (pipeline overview)](#how-it-works-pipeline-overview)
- [Modeling choices](#modeling-choices)
- [Testing & Evaluation](#testing--evaluation)
- [Contributing](#contributing)
- [License](#license)

## About
This repository demonstrates a reproducible pipeline for disease prediction based on a large symptom → disease dataset. The pipeline trains multiple machine learning models (sklearn-based) and exposes an easy-to-use Flask web interface for inference. The project is intended for educational and research purposes (student project).

## Features
- Reproducible training pipeline (data preprocessing → model training → model serialization)
- Multiple model types supported (example: Random Forest, Naive Bayes, Decision Tree, Neural Network, XGBoost)
- Flask-based REST + simple web UI for model inference
- Clean project layout and `requirements.txt` to reproduce environment
- MIT license included

## Project Structure
```
DiseasePredictionAI/
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
├── src/
│   ├── __pycache__/
│   ├── models/
│   ├── static/
│   ├── templates/
│   ├── app.py
│   ├── preprocess.py
│   └── train_models.py
└── venv/
```
> **Note:** The `data/` folder is intentionally not tracked (too large). See "Dataset & Credits" below to download it.

## Dataset & Credits
The dataset used to train the models is **not** included in the repository due to its size. It can be downloaded from Mendeley Data (please observe the dataset license and citation requirements):

**Dataset:** _Disease and symptoms dataset 2023_ — Mendeley Data.  
Download link: `https://data.mendeley.com/datasets/2cxccsxydc/1`

**Credit / Contributor:** Bran Stark. Licensed under **CC BY 4.0** (see dataset page for details).

> Please cite / credit the dataset and observe the CC BY 4.0 license when using or publishing results derived from it.

## Requirements
- Python 3.8+
- See `requirements.txt` for specific package versions. Typical packages:
  - Flask
  - scikit-learn
  - pandas
  - numpy
  - joblib (model serialization)
  - xgboost (optional, if used)
  - matplotlib (optional for plots)

## Installation (quickstart)
1. Clone the repo
```bash
git clone https://github.com/efimijacuneva/DiseasePredictionAI.git
cd DiseasePredictionAI
```

2. Create a virtual environment and install dependencies
```bash
python -m venv venv
# activate the venv (Windows)
venv\\Scripts\\activate
# OR (mac / linux)
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

3. Download the dataset and place it under `data/` (see Dataset & Credits)
```bash
# example (replace with actual download/placement commands)
# download and unzip the dataset into the repository data/ folder
# e.g. data/dataset.csv or similar file expected by src/preprocess.py
```

## Usage

### Training the models (pipeline)
You **must** run the training script first to produce serialized model artifacts under `models/`.
```bash
# from repository root
python src/train_models.py
```
What `train_models.py` does (typical pipeline behavior):
1. `src/preprocess.py` reads `data/` and returns processed feature matrix and labels.
2. Training code performs feature engineering, encoding, and model training.
3. Trained models and supporting artifacts (LabelEncoders, feature lists, scalers) are saved into `models/` (e.g. `models/trained_models.pkl` or per-model `.joblib` files).

> If your `train_models.py` accepts CLI args (e.g. `--subset`, `--outdir`, `--seed`), use them as documented in the script. The command above assumes default settings in the repository.

### Running the Flask app (inference)
After you have trained and saved models into `models/`, start the Flask app for inference:
```bash
python app.py
```
Open your browser to `http://127.0.0.1:5000/` to use the UI or call API endpoints (see app docs / routes).

## How it works (pipeline overview)
1. **Preprocess**: `src/preprocess.py` — loads raw CSV(s), validates symptom columns, merges rare diseases if needed, encodes labels and features.
2. **Train**: `src/train_models.py` — trains the chosen estimators, evaluates them (train/validation metrics), and saves artifacts to `models/`.
3. **Serve**: `app.py` — loads the saved artifacts and exposes a web UI / REST endpoint for inference.
4. **Evaluate**: optional notebooks / scripts compute precision/recall/F1 and confusion matrices.

## Modeling choices
- Use stratified sampling when training to ensure rare diseases are not completely omitted from validation sets.
- Serialize sklearn pipelines + models using `joblib` or `pickle` (note sklearn compatibility warnings when moving across major versions — re-train or re-serialize if upgrading scikit-learn).
- Consider class imbalance techniques for rare labels (class weighting, sampling, hierarchical predictions).

## Testing & Evaluation
- Unit tests for preprocessing functions and model input validation (recommended).
- Evaluate models using stratified cross-validation and report macro/micro F1, precision, recall.
- Keep a `notebooks/` folder for exploratory analysis and reproducible evaluation plots.

## Contributing
Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request on GitHub. When contributing:
- Add tests for new functionality
- Keep changes small and well-documented
- Update `requirements.txt` if you add packages

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

*Prepared by Efimija Cuneva*  
*July 2025*  

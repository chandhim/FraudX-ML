# FraudX: ML Fraud Detection

FraudX detects fraudulent mobile-money transactions with a machine learning model. It has two parts:

- **`analysis.ipynb`**: exploratory data analysis and model training. Saves the trained pipeline to `fraud_detection_model.pkl`.
- **`fraud_detection.py`**: a Streamlit web app that scores a single transaction with the saved model and shows the result, risk level and money flow.

## Table of Contents
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the Web App](#running-the-web-app)
- [Running the Notebook](#running-the-notebook)
- [Running the Tests](#running-the-tests)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)

## Dataset

The data is the **PaySim** simulated mobile-money dataset from Kaggle: about 6.3 million transactions, of which 0.13% are fraudulent. It is synthetic, so amounts are simulated units rather than a real currency.

- **Link:** [Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset)

The dataset is only needed to run the notebook. The web app uses the committed model and works without it.

1. Download the dataset (you may need a Kaggle account and to accept the dataset terms).
2. Rename the CSV file to `creditcard.csv`. The notebook reads it under that name.
3. Put it in the project root. CSV files are git-ignored, so it won't be committed by accident.

## Prerequisites

- Python 3.10 or newer (required by scikit-learn 1.7)
- pip
- Git (optional, for cloning the repository)

## Setup

Create and activate a virtual environment:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

That is enough to run the app. To also run the notebook and the tests:

```bash
pip install -r requirements-dev.txt
```

## Running the Web App

```bash
streamlit run fraud_detection.py
```

The app opens at http://localhost:8501.

- **Try a sample:** one click loads a ready-made transaction (an everyday payment, a cash deposit, or two typical fraud patterns) and scores it.
- **Or enter your own:** pick the type and amount; the new balances are calculated from the amount unless you switch that off to enter them by hand. Then click **Analyze transaction**.
- **Result:** the verdict and fraud score, a chart of what drove the score (each input's contribution to the model's log-odds), any data checks, and the balance details.
- The sidebar explains the dataset, the model and its test-set recall and precision.

The app checks the input before scoring:

- **Errors** (negative values, a zero amount) stop the prediction.
- **Warnings** (balances that don't add up, a large transfer or cash-out) are shown, but the transaction is still scored. Inconsistent balances are themselves a fraud signal in this data.

## Running the Notebook

```bash
jupyter notebook analysis.ipynb
```

Run the cells in order from the top. The notebook loads the dataset, explores it, trains a logistic regression pipeline and saves it as `fraud_detection_model.pkl`, overwriting the committed model.

If you retrain with a different scikit-learn version, update the `scikit-learn` pin in `requirements.txt` to match.

## Running the Tests

```bash
pytest
```

The tests cover the input validation rules and run the app headlessly to check that Predict and Reset work. They also run automatically on GitHub Actions for every push and pull request.

## Project Structure

```
FraudX/
├── fraud_detection.py         # Streamlit app
├── validation.py              # input validation rules used by the app
├── fraud_detection_model.pkl  # trained pipeline (scikit-learn 1.7.2)
├── analysis.ipynb             # EDA and model training
├── images/
│   ├── FraudX.svg             # logo used by the app
│   └── FraudX.png             # logo (PNG)
├── tests/                     # pytest suite
├── requirements.txt           # app dependencies (pinned)
├── requirements-dev.txt       # notebook and test dependencies
├── .github/workflows/         # CI
├── .streamlit/config.toml     # app theme
└── creditcard.csv             # dataset (download separately, not committed)
```

## Troubleshooting

**`ModuleNotFoundError`**
Make sure your virtual environment is active and you installed `requirements.txt` (or `requirements-dev.txt` for the notebook).

**`InconsistentVersionWarning` when the app loads the model**
Your scikit-learn version differs from the one that saved the model. Run `pip install -r requirements.txt` to get the pinned version.

**Notebook can't find the dataset**
Check that the CSV is named `creditcard.csv` and is in the project root.

**Out of memory in the notebook**
The dataset is large (6.3M rows). Close other applications or use a machine with more RAM.

## Notes

- The model was trained with scikit-learn 1.7.2. Loading it with another version may warn or give wrong results.
- Only load `.pkl` model files you trust: unpickling can run arbitrary code.
- Amounts are simulated units, so the 50,000 high-risk threshold in the data checks is illustrative only.
- The current model flags most real frauds but also many legitimate transactions (low precision), so treat a FRAUDULENT result as "needs review", not a final verdict.

## License

The dataset is subject to its own terms on Kaggle; see the dataset page.

## Acknowledgments

- Dataset: [Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset)

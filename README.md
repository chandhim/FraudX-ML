# ML Fraud Detection Project

A machine learning project for detecting fraudulent transactions using classification algorithms.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Dataset Download](#dataset-download)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)

## Overview

This project implements machine learning models to detect fraudulent credit card transactions. The analysis is performed using Python and popular data science libraries in a Jupyter notebook environment.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
- Python 3.8 or higher
- pip (Python package manager)
- Jupyter Notebook or JupyterLab
- Git (optional, for cloning the repository)

## Environment Setup

### Step 1: Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

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

### Step 2: Upgrade pip

```bash
python -m pip install --upgrade pip
```

## Dataset Download

1. Download the fraud detection dataset from Kaggle:
   - **Dataset Link:** [Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download)
   
2. You may need to:
   - Create a Kaggle account if you don't have one
   - Accept the dataset terms and conditions
   
3. After downloading, place the `creditcard.csv` file in the project root directory:
   ```
   ML-fraud-detection/
   ├── creditcard.csv  <- Place the dataset here
   ├── analysis.ipynb
   └── README.md
   ```

## Installation

Install all required dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter notebook
```

### Required Libraries

The project uses the following Python libraries:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms and tools
- **jupyter**: Interactive notebook environment
- **notebook**: Jupyter notebook interface

### Optional: Using requirements.txt

If a `requirements.txt` file is provided, you can install all dependencies at once:

```bash
pip install -r requirements.txt
```

## Running the Project

### Step 1: Launch Jupyter Notebook

Make sure your virtual environment is activated, then start Jupyter Notebook:

```bash
jupyter notebook
```

This will open Jupyter in your default web browser.

### Step 2: Open the Analysis Notebook

1. In the Jupyter interface, navigate to the project directory
2. Click on `analysis.ipynb` to open the notebook
3. Run the cells sequentially by pressing `Shift + Enter` or using the "Run" button

### Step 3: Execute the Analysis

- Start from the first cell and run each cell in order
- The notebook will load the dataset, perform exploratory data analysis, and train machine learning models
- Review the outputs, visualizations, and model performance metrics

## Project Structure

```
ML-fraud-detection/
├── analysis.ipynb      # Main Jupyter notebook with ML analysis
├── creditcard.csv      # Dataset (download separately)
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## Troubleshooting

### Common Issues

**Issue: Module not found error**
- Solution: Ensure all dependencies are installed in your active virtual environment

**Issue: Dataset not found**
- Solution: Verify that `creditcard.csv` is in the project root directory

**Issue: Jupyter notebook won't start**
- Solution: Try reinstalling Jupyter: `pip install --upgrade jupyter notebook`

**Issue: Out of memory errors**
- Solution: Close other applications or consider using a machine with more RAM

## Notes

- Make sure to activate your virtual environment every time you work on the project
- The dataset file (`creditcard.csv`) is typically large and should not be committed to version control
- Results may vary depending on the random state and model parameters used

## License

Please refer to the dataset license on Kaggle for usage terms and conditions.

## Acknowledgments

- Dataset provided by: [Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset)

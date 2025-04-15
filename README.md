# Weekly Sales Machine Learning Models

---

## Overview

This project predicts weekly sales for retail stores using machine learning models. It includes data preprocessing, feature engineering, and training/testing of models such as Random Forest and Gradient Boosting Regressors.

---

## Features

- **Data Preprocessing**: Handles missing values and converts date columns to datetime format.
- **Feature Engineering**: Adds holiday and date-based features to improve model performance.
- **Model Training**: Implements Random Forest and Gradient Boosting Regressors with hyperparameter tuning.
- **Evaluation**: Calculates metrics like Weighted MAE, RMSE, and R² for model performance.
- **Predictions**: Outputs final predictions for test datasets.

---

## Technologies Used

- Programming Language: Python

- Libraries & Frameworks:
    - [Pandas](https://pandas.pydata.org): Data manipulation
    - [NumPy](https://numpy.org): Numerical operations
    - [Scikit-learn](https://scikit-learn.org/stable/): Machine learning model implementation
    - [Matplotlib](https://matplotlib.org): Data visualisation
    - [Seaborn](https://seaborn.pydata.org): Data visualisation

- Tools:
    - [Git](https://git-scm.com): Version control
    - [Jupyter Notebook](https://jupyter.org): Interactive development
    - [VS Code](https://code.visualstudio.com): Final development and README file development

---

## Project Structure
```
CW2_code<br/>
├── CW2_code.py # Main Python script containing the implementation<br/>
├── train.csv # Training dataset<br/>
├── test.csv # Test dataset<br/>
├── stores.csv # Stores data<br/>
├── features.csv # Additional features for the datasets<br/>
├── README.md # Project documentation<br/>
├── Report.docx # Detailed project report<br/>
```
---

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git (for cloning the repository)

### Installation

1. Clone the repository:

    ```
    git clone https://github.com/Mattytomo365/Weekly-Sales-Machine-Learning-Models.git
    cd Weekly-Sales-Machine-Learning-Models
    ```

2. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

### Usage

1. Place the datasets (`train.csv`, `test.csv`, `stores.csv`, `features.csv`) in the project directory.

2. Run the main script:

    ```
    python CW2_code.py
    ```

3. View the outputs, including the evaluation metrics and predictions.

---

## Contributing

**Contributions are welcome!**\
Please fork the repository and submit a pull request with your changes.

---

## Contact

For any questions or feedback, feel free to reach out:
- **Email:** matty.tom@icloud.com
- **GitHub:** [Mattytomo365](https://github.com/Mattytomo365)




# Weekly Sales Machine Learning Models

---

## Overview

This project consists of two machine learning models (Random Forest Regressor & Gradient Boosting Regressor), these models aim to predict the weekly sales for a large retail company, taking the affects of holiday periods into effect. Both models used are evaluated, tuned for optimal performance and compared. Both models are trained using historical data contained in datasets.

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
CW2_code
├── CW2_code.py         # Main Python script containing the implementation
├── train.csv           # Training dataset
├── test.csv            # Test dataset
├── stores.csv          # Stores data
├── features.csv        # Additional features for the datasets
├── README.md           # Project documentation
├── Report.docx         # Detailed project report
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
    ```

2. Navigate to the project directory:

    ```
    cd Weekly_Sales_Models
    ```


3. Download necessary resources.

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




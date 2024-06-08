# Automobile Price Prediction Using Neural Networks and Cross-Validation

## Overview
This project focuses on predicting automobile prices using neural networks. The data is pre-processed and various features are encoded to create a robust model. The model's performance is evaluated using cross-validation techniques.

## Context
In this project, we aim to predict the prices of automobiles by building and training a neural network model. The data undergoes significant pre-processing to handle inconsistencies and encode categorical variables appropriately. The neural network model is then trained, and its performance is evaluated using cross-validation.

## Objectives:
- Develop a neural network model to predict automobile prices.
- Pre-process the dataset to handle missing values and encode categorical features.
- Use cross-validation to evaluate the model's performance.

## Steps Involved:

### 1. Data Pre-processing:
- Load the dataset using Pandas.
- Drop irrelevant columns (`name`, `dateCrawled`, `dateCreated`, `nrOfPictures`, `postalCode`, `lastSeen`).
- Handle missing values by replacing them with the most frequent values.
- Encode categorical variables using `LabelEncoder` and `OneHotEncoder`.

### 2. Building the Neural Network:
- Create a neural network using Keras with three layers:
  - Input layer with 158 units and ReLU activation.
  - Hidden layer with 158 units and ReLU activation.
  - Output layer with 1 unit and linear activation.

### 3. Model Training and Evaluation:
- Train the model using Keras' `KerasRegressor` with 100 epochs and a batch size of 300.
- Evaluate the model using cross-validation (`cross_val_score` with 10 folds) and calculate the mean and standard deviation of the mean absolute error.

## Files:

- **autos_regressao.py**: Contains the code for data pre-processing and model training.
- **autos-regressao_Crossr.py**: Contains the code for cross-validation of the model.

## How to Run:
1. Clone the repository:
    ```sh
    git clone <repository URL>
    ```

2. Navigate to the project directory:
    ```sh
    cd <repository name>
    ```

3. Ensure you have the required libraries installed. You can install them using:
    ```sh
    pip install pandas keras scikit-learn
    ```

4. Run the scripts to see the results:
    ```sh
    python autos_regressao.py
    python autos-regressao_Crossr.py
    ```

## Contribution
Contributions are welcome! Feel free to open issues and submit pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

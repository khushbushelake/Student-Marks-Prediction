# Student Marks Prediction Using Machine Learning

## Project Overview
This project predicts student marks based on the number of study hours using Linear Regression, a supervised machine learning algorithm. It demonstrates model training, prediction, evaluation, and visualization using Python.

## Objectives
- Understand supervised machine learning.
- Implement Linear Regression using Scikit-learn.
- Predict marks based on study hours.
- Evaluate model performance.
- Visualize the relationship between study hours and marks.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Algorithm
Linear Regression

The model learns the relationship between study hours (input) and marks (target) using a sample dataset.

## Features
- Loads data from a CSV file.
- Splits data into training and testing sets.
- Trains a Linear Regression model.
- Predicts marks based on user input.
- Evaluates performance using MAE, MSE, and R² Score.
- Generates a graph of actual data and the regression line.

## Project Structure
    Student-Marks-Prediction/
    ├── app.py
    ├── student_data.csv
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    └── student_marks_prediction.png

## Installation and Setup

### 1. Clone the repository
    git clone https://github.com/khushbushelake/Student-Marks-Prediction.git

### 2. Open the project folder
    cd Student-Marks-Prediction

### 3. Install dependencies
    python -m pip install -r requirements.txt

### 4. Run the project
    python app.py

On Windows, you can also use:
    py app.py

## How It Works
1. Loads the dataset using Pandas.
2. Separates study hours and marks into input and target variables.
3. Splits the data into training and testing sets.
4. Trains the Linear Regression model.
5. Evaluates the model using MAE, MSE, and R² Score.
6. Accepts study hours from the user.
7. Predicts the expected marks.
8. Generates a graph named `student_marks_prediction.png`.

## Sample Prediction
Input:
    Enter the number of study hours: 7

Expected prediction with the supplied sample dataset:
    Predicted Marks: approximately 70.00

The actual evaluation metrics are displayed when the program runs.

## Limitations
- Uses a small, synthetic dataset for educational purposes.
- The sample data follows a nearly perfect linear relationship.
- Actual academic performance depends on multiple factors beyond study hours.
- Predictions outside the dataset range may be unreliable.

## Future Improvements
- Use a larger and more representative dataset.
- Include attendance and previous marks as additional features.
- Develop an interactive interface using Streamlit.
- Compare different regression algorithms.
- Improve evaluation using cross-validation.

## Author
Khushbu Shelake

GitHub: https://github.com/khushbushelake/Student-Marks-Prediction

## Purpose
This project was developed for educational purposes to demonstrate the implementation of a machine learning model using Python.

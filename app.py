import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------
# 1. Load the dataset
# ------------------------------------------

data = pd.read_csv("student_data.csv")

print("=" * 45)
print(" STUDENT MARKS PREDICTION SYSTEM")
print("=" * 45)

print("\nDataset Preview:")
print(data.head())

print("\nDataset Information:")
print(data.info())

# ------------------------------------------
# 2. Prepare the data
# ------------------------------------------

X = data[["Hours"]]
y = data["Marks"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------
# 3. Train the Linear Regression model
# ------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# ------------------------------------------
# 4. Evaluate the model
# ------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.4f}")

print(f"\nRegression Coefficient: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# ------------------------------------------
# 5. Predict marks for new study hours
# ------------------------------------------

try:
    hours = float(input("\nEnter the number of study hours: "))

    if hours < 0:
        print("Study hours cannot be negative.")

    else:
        predicted_marks = model.predict(
            pd.DataFrame({"Hours": [hours]})
        )[0]

        print("\nPrediction Result")
        print("-" * 30)
        print(f"Study Hours: {hours}")
        print(f"Predicted Marks: {predicted_marks:.2f}")

        if predicted_marks > 100:
            print(
                "Note: This prediction is outside the "
                "0-100 marks range."
            )

except ValueError:
    print("Invalid input! Please enter a numeric value.")

# ------------------------------------------
# 6. Visualize the dataset and model
# ------------------------------------------

plt.figure(figsize=(9, 5))

plt.scatter(
    data["Hours"],
    data["Marks"],
    color="blue",
    label="Actual Data"
)

plt.plot(
    data["Hours"],
    model.predict(data[["Hours"]]),
    color="red",
    label="Linear Regression Line"
)

plt.title("Student Marks Prediction")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("student_marks_prediction.png")
plt.show()

print("\nGraph saved as student_marks_prediction.png")
print("\nProject completed successfully!")
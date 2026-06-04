import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# LOAD DATASET

df = pd.read_csv("dataset/cleaned_titanic.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# SELECT FEATURES

# Target variable
y = df["Survived"]

# Remove target column
X = df.drop(columns=["Survived"])

# Keep only numerical columns
X = X.select_dtypes(include=["int64", "float64", "bool"])

print("\nFeatures Used:")
print(X.columns)

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Size:", len(X_train))
print("Testing Size:", len(X_test))


# LINEAR REGRESSION


lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)

print("\n========== LINEAR REGRESSION RESULTS ==========")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)


# DECISION TREE


dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=4
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_dt)

print("\n========== DECISION TREE RESULTS ==========")
print("Accuracy:", accuracy)


# GRAPH 1
# LINEAR REGRESSION RESULTS

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred_lr)

plt.xlabel("Actual Survival")
plt.ylabel("Predicted Survival")

plt.title("Linear Regression Predictions")

plt.savefig(
    "outputs/linear_regression_results.png"
)

plt.show()

 
# GRAPH 2
# DECISION TREE

plt.figure(figsize=(15, 8))

plot_tree(
    dt_model,
    filled=True,
    feature_names=X.columns,
    class_names=["Not Survived", "Survived"]
)

plt.title("Decision Tree Visualization")

plt.savefig(
    "outputs/decision_tree.png"
)

plt.show()

# GRAPH 3
# MODEL COMPARISON

plt.figure(figsize=(8, 5))

models = ["Linear Regression", "Decision Tree"]
scores = [r2, accuracy]

bars = plt.bar(models, scores)

plt.ylabel("Score")
plt.title("Model Performance Comparison")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.3f}",
        ha="center"
    )

plt.savefig(
    "outputs/model_comparison.png"
)

plt.show()

print("\nTask 3 Completed Successfully!!!")
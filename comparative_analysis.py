import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = np.array([
    [101,  8,  92,  85, 1],
    [102,  3,  68,  55, 0],
    [103,  7,  88,  78, 1],
    [104,  2,  60,  45, 0],
    [105,  9,  95,  91, 1],
    [106,  4,  72,  62, 0],
    [107,  6,  85,  74, 1],
    [108,  1,  55,  38, 0],
    [109, 10,  96,  94, 1],
    [110,  5,  78,  68, 1],
    [111,  2,  65,  50, 0],
    [112,  8,  90,  82, 1],
    [113,  3,  70,  58, 0],
    [114,  7,  87,  80, 1],
    [115,  4,  75,  60, 0]
])

dataframe = pd.DataFrame(data, columns=["Student_ID", "Study_Hours", "Attendance", "Assignment", "Result"])


dataframe.to_csv("Student_Dataset.csv", index=False)

df = pd.read_csv("Student_Dataset.csv")
print(df.describe)

df.groupby("Result")[["Study_Hours","Attendance","Assignment"]].mean()

plt.figure(figsize=(7,4))
plt.scatter(df["Study_Hours"], df["Attendance"], c=df["Result"])
plt.xlabel("Study hours")
plt.ylabel("Attendance")
plt.title("Student performance pattern")
plt.show()


X = df[["Study_Hours", "Attendance", "Assignment"]]
y = df["Result"]



print("X shape:", X.shape)
print("y shape:", y.shape)



X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

print("Model trained!")


predictions = model.predict(X_test)

print("Predictions:", predictions)
print("X_test:", X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Test accuracy: {accuracy:.2%}")

plt.figure(figsize=(13,7))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True
)
plt.title("What did the model learn?")
plt.show()
# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize weights and bias
2.Choose a loss function
3.Set hyperparameters (learning rate, regularization, etc.)
4.Shuffle training data
5.For each epoch:
6.For each training sample:
7.Compute gradient of loss
8.Update weights using gradient descent
9.Apply regularization
10.Check for convergence or stop after max iterations

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SHAKTHI BALAN V
RegisterNumber:  212225230259
*/
```
```
"""
Program to implement the prediction of iris species using SGD Classifier.
Developed by: PRIYAN M & PRANAV S
Register Numbers: 25017302 | 212224040242
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


iris = load_iris()


df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("First 5 rows of the dataset:")
print(df.head())


X = df.drop('target', axis=1)
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train)


y_pred = sgd_clf.predict(X_test)


print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Oranges',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.title("Confusion Matrix — Iris SGD Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
```

## Output:
<img width="766" height="618" alt="image" src="https://github.com/user-attachments/assets/a4727679-80d0-4a99-aac8-8a712888a4db" />
<img width="748" height="540" alt="image" src="https://github.com/user-attachments/assets/9efce6b3-7465-461d-a29c-ff00797d9a28" />




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

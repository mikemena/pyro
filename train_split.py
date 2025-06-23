from sklearn.model_selection import train_test_split
import numpy as np

X = np.array([[1,2,3], [3,4,0],[5,6,0], [7,8,0], [9,10,0]]) # Features
y = np.array([0,1,0,1,0]) # Lables

#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

print("Training features:\n", X_train)
print("Test features:\n", X_test)
print("Training labels:\n", y_train)
print("Test labels:\n", y_test)

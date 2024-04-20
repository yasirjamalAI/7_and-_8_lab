from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Introduce some random noise into the dataset
import numpy as np
np.random.seed(42)
X_noise = X + np.random.normal(0, 0.5, size=X.shape)

# Split the noisy dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_noise, y, test_size=0.2, random_state=42)

# Initialize a Support Vector Machine classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Train the classifier on the training set
svm_classifier.fit(X_train, y_train)

# Predict labels for the test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)*100
print("Accuracy:", accuracy,"%")

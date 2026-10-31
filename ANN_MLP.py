#----------------------------------------------------------
# ANN Classification using MLPClassifier
# Dataset: Breast Cancer Wisconsin (Built-in Dataset)
#----------------------------------------------------------
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#----------------------------------------------------------
# 1. Load the dataset
#----------------------------------------------------------
data = load_breast_cancer()

X = data.data
y = data.target
#----------------------------------------------------------
# 2. Display dataset information
#----------------------------------------------------------
print("Dataset Shape:", X.shape)
print("Features:", data.feature_names)
print("Classes:", data.target_names)
#----------------------------------------------------------
# 3. Split the dataset
#----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
#----------------------------------------------------------
# 4. Feature scaling
#----------------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#----------------------------------------------------------
# 5. Create the ANN model
#----------------------------------------------------------
model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
#----------------------------------------------------------
# 6. Train the model
#----------------------------------------------------------
model.fit(X_train, y_train)
#----------------------------------------------------------
# 7. Make predictions
#----------------------------------------------------------
y_pred = model.predict(X_test)
#----------------------------------------------------------
# 8. Evaluate the model
#----------------------------------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))
#----------------------------------------------------------
# 9. Test the model with a sample input
#----------------------------------------------------------
sample = X_test[0].reshape(1, -1)

prediction = model.predict(sample)

print("\n--- Model Testing ---")
print("Predicted Class:", data.target_names[prediction[0]])
print("Actual Class:", data.target_names[y_test[0]])

probabilities = model.predict_proba(sample)

print("\nPrediction Probabilities:")
for class_name, probability in zip(data.target_names, probabilities[0]):
    print(f"{class_name}: {probability:.4f}")
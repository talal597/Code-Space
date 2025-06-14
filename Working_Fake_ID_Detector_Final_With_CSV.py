
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\Naser\Downloads\archive\Instagram_fake_profile_dataset.csv")

# Feature selection
X = df.drop(columns=['fake'])
y = df['fake']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Output performance
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Save prediction results
results_df = pd.DataFrame(X_test, columns=df.columns[:-1])
results_df['Actual'] = y_test.values
results_df['Predicted'] = y_pred
results_df['Fake_Probability'] = y_proba
results_df.to_csv(r"C:\Users\Naser\Downloads\Fake_ID_Detection_Results.csv", index=False)

# Plot and show feature importance
feature_importances = model.feature_importances_
feature_names = df.columns[:-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

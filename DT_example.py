import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Create a sample dataset
data = {
    'soreThroat': ['yes', 'no', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'no', 'yes'],
    'fever': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'yes'],
    'swollen': ['yes', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'no'],
    'congestion': ['yes', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes'],
    'headache': ['yes', 'yes', 'no', 'no', 'no', 'no', 'no', 'yes', 'yes', 'yes'],
    'disease': ['Strep throat', 'Allergy', 'Cold', 'Strep throat', 'Cold', 'Allergy', 'Strep throat', 'Allergy', 'Cold',
                'Cold']
}

df = pd.DataFrame(data)

# Convert categorical features to numerical
for col in ['soreThroat', 'fever', 'swollen', 'congestion', 'headache']:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Split the data into features and target
X = df[['soreThroat', 'fever', 'swollen', 'congestion', 'headache']]
y = df['disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, class_names=df['disease'].unique(), filled=True)
plt.title('Decision Tree')
plt.show()

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
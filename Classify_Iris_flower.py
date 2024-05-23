import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris_db = load_iris()
X = iris_db.data
y = iris_db.target

# Encode the target variable to one-hot format
y = keras.utils.to_categorical(pd.Categorical(y).codes)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(x_train.shape[1],)),
    keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=32)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Accuracy :', test_acc)
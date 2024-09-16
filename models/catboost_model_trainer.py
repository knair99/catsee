from catboost import CatBoostClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Dummy example with the digits dataset for illustration
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

model = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.1, loss_function='MultiClass')
model.fit(X_train, y_train)

# Save model
model.save_model('catboost_model.cbm')
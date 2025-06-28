from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from data_preprocessing import load_and_split, make_preprocessor

X_train, X_test, y_train, y_test = load_and_split()
preprocessor = make_preprocessor()

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", DecisionTreeClassifier(random_state=42))
])

param_grid = {
    "clf__max_depth": [3, 5, 10],
    "clf__min_samples_split": [2, 5, 10],
    "clf__criterion": ["gini", "entropy"]
}

grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)

print("🔎 Best Parameters:", grid.best_params_)
print("\n📋 Classification Report:\n", classification_report(y_test, grid.predict(X_test)))
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from data_preprocessing import load_and_split, make_preprocessor
from scipy.stats import randint

X_train, X_test, y_train, y_test = load_and_split()
preprocessor = make_preprocessor()

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(random_state=42))
])

param_dist = {
    "clf__n_estimators": randint(50, 200),
    "clf__max_depth": [None, 10, 20, 30],
    "clf__min_samples_split": randint(2, 10),
    "clf__criterion": ["gini", "entropy"]
}

random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=20, cv=5, random_state=42)
random_search.fit(X_train, y_train)

print("🎯 Best Parameters:", random_search.best_params_)
print("\n📋 Classification Report:\n", classification_report(y_test, random_search.predict(X_test)))
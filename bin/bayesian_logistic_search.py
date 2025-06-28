from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from skopt import BayesSearchCV
from data_preprocessing import load_and_split, make_preprocessor

X_train, X_test, y_train, y_test = load_and_split()
preprocessor = make_preprocessor()

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(
        solver="saga",
        max_iter=5000, class_weight="balanced"
    ))
])

param_space = {
    "clf__C": (1e-3, 100.0, "log-uniform"),
    "clf__penalty": ["l2", "l1"]
}

search = BayesSearchCV(
    estimator=pipeline,
    search_spaces=param_space,
    n_iter=20,
    cv=5,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

print("🔎 Best Params:", search.best_params_)
print("\n📋 Classification Report:\n", classification_report(y_test, search.predict(X_test)))
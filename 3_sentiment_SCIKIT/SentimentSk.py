# Sentiment with scikit-learn: TF-IDF (transformer) + Logistic Regression
# ---------------------------------------------------------------
# Requirements: scikit-learn, joblib (usually comes with sklearn)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib

# 1) Example data (replace with your own)
X = [
    "Loved it! What a fantastic movie.",
    "Absolutely terrible. Waste of time.",
    "Great acting and plot. Highly recommended!",
    "I hated every minute of it.",
    "It was okay, not great but not bad either.",
    "Brilliant direction and beautiful soundtrack.",
    "Awful experience, I left the theater early.",
    "Mediocre at best.",
    "One of the best films this year!",
        "Not for me."
]
# Binary labels: 1 = positive, 0 = negative (tweak to your needs)
y = np.array([1,0,1,0,1,1,0,0,1,0])

# 2) Train/validation split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3) Build a Pipeline: TF-IDF (transformer) -> Logistic Regression (classifier)
pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=200, n_jobs=None))  # n_jobs not used by liblinear, set solver below
])

# (Optional) Handle class imbalance (if any)
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
cw_dict = {c: w for c, w in zip(classes, class_weights)}

# 4) Hyperparameter search (small grid to keep it fast)
param_grid = {
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__min_df": [1, 2],
    "tfidf__max_df": [0.9, 1.0],
    "tfidf__max_features": [None, 5000],
    "clf__solver": ["liblinear", "lbfgs"],  # liblinear good for small/medium; lbfgs supports L2
    "clf__C": [0.25, 1.0, 4.0],
    "clf__class_weight": [cw_dict, None],
}
search = GridSearchCV(
    pipe,
    param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=0
)
search.fit(X_train, y_train)

print("Best params:", search.best_params_)
print("Best CV accuracy:", round(search.best_score_, 4))

# 5) Evaluate on test set
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nTest accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 6) Use the model
samples = [
    "What a masterpiece! I cried twice.",
    "This was unwatchable and boring."
]
pred = best_model.predict(samples)
proba = best_model.predict_proba(samples)  # probability for each class
for text, p, pr in zip(samples, pred, proba):
    print(f"\nText: {text}\nPredicted: {'positive' if p==1 else 'negative'} | Probabilities: {pr}")

# 7) Persist model to disk
joblib.dump(best_model, "sentiment_lr_tfidf.joblib")
print("\nSaved model to sentiment_lr_tfidf.joblib")

# 8) Load later and predict
loaded = joblib.load("sentiment_lr_tfidf.joblib")
print("Loaded model predicts:", loaded.predict(["I loved the cinematography."]))

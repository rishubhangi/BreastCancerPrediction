import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)

# 1. Load and clean (same steps as the project)
data = pd.read_csv("data/data.csv")
data = data.drop(["Unnamed: 32", "id"], axis=1)
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# 2. Split FIRST, then scale inside a pipeline (no data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
fig_roc, ax_roc = plt.subplots()
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "CV accuracy (mean)": round(cv_scores.mean(), 4),
        "CV accuracy (std)": round(cv_scores.std(), 4),
        "Test accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, prob), 4),
    })
    print(name, "confusion matrix [[TN FP][FN TP]]:\n", confusion_matrix(y_test, pred))
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Benign", "Malignant"], ax=ax)
    ax.set_title(name)
    fig.savefig(f"confusion_{name.replace(' ', '_')}.png", dpi=200, bbox_inches="tight")
    RocCurveDisplay.from_predictions(y_test, prob, name=name, ax=ax_roc)

ax_roc.set_title("ROC curves")
fig_roc.savefig("roc_curves.png", dpi=200, bbox_inches="tight")
results = pd.DataFrame(rows)
print(results.to_string(index=False))
results.to_csv("model_comparison.csv", index=False)
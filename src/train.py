"""Train several models, keep the best one (by ROC-AUC) and save it."""
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import evaluation as ev
from src.data_preprocessing import (ROOT, build_preprocessor, clean_data,
                                    load_data, prepare_xy, split_data)

MODEL_PATH = ROOT / "models" / "churn_model.pkl"


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def train(save=True):
    raw = load_data()
    X, y = prepare_xy(raw)
    X_train, X_test, y_train, y_test = split_data(X, y)

    results, fitted = {}, {}
    for name, clf in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        results[name] = ev.compute_metrics(y_test, pred, prob)
        fitted[name] = (pipe, pred, prob)
        print(f"{name:<22} AUC={results[name]['roc_auc']:.4f}  "
              f"F1={results[name]['f1']:.4f}  Acc={results[name]['accuracy']:.4f}")

    best = max(results, key=lambda n: results[n]["roc_auc"])
    pipe, pred, prob = fitted[best]
    print(f"\nBest model: {best}")

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)
        clean = clean_data(raw)
        ev.plot_churn_distribution(clean)
        ev.plot_correlation_matrix(clean)
        ev.plot_confusion_matrix(y_test, pred)
        ev.plot_roc_curve(y_test, prob)
        ev.write_metrics(results, best)
        print(f"Saved model to {MODEL_PATH}")
    return pipe, results, best


if __name__ == "__main__":
    train()

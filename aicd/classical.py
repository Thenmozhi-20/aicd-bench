"""Classical baselines: Logistic Regression, SVM, CatBoost x {TF-IDF, AST, both}; plus random/majority."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .features import FeatureBuilder


def make_model(name, n_classes):
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == "svm":
        return make_pipeline(StandardScaler(), LinearSVC())
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=500, verbose=100,
                                  loss_function="Logloss" if n_classes == 2 else "MultiClass")
    raise ValueError(name)


def run_classical(train, test, y_train, n_classes, model_name, feats):
    fb = FeatureBuilder(feats)
    Xtr = fb.fit_transform(train)
    Xte = fb.transform(test)
    m = make_model(model_name, n_classes)
    m.fit(Xtr, y_train)
    return np.asarray(m.predict(Xte)).reshape(-1).astype(int)


def dummy_baselines(y_train, n_test, n_classes, seed=0):
    rng = np.random.default_rng(seed)
    counts = np.bincount(y_train, minlength=n_classes)
    return {"random": rng.choice(n_classes, size=n_test, p=counts / counts.sum()),
            "majority": np.full(n_test, counts.argmax())}

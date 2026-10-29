"""Classical baselines: Logistic Regression, SVM, CatBoost x {TF-IDF, AST, both}; plus random/majority."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .features import FeatureBuilder


def make_model(name, n_classes):
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
    if name == "svm":
        return make_pipeline(StandardScaler(), LinearSVC(dual=False, max_iter=2000, tol=1e-3, random_state=0))
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300 if n_classes > 2 else 500, verbose=100, random_seed=0, thread_count=-1,
                                  loss_function="Logloss" if n_classes == 2 else "MultiClass")
    raise ValueError(name)


def run_classical(train, test, y_train, n_classes, model_name, feats, return_model=False):
    fb = FeatureBuilder(feats)
    Xtr = fb.fit_transform(train)
    Xte = fb.transform(test)
    m = make_model(model_name, n_classes)
    m.fit(Xtr, y_train)
    pred = np.asarray(m.predict(Xte)).reshape(-1).astype(int)
    if return_model:
        return pred, m, fb
    return pred


def dummy_baselines(y_train, n_test, n_classes, seed=0):
    rng = np.random.default_rng(seed)
    counts = np.bincount(y_train, minlength=n_classes)
    return {"random": rng.choice(n_classes, size=n_test, p=counts / counts.sum()),
            "majority": np.full(n_test, counts.argmax())}


def explain_model_features(model, fb, top_k=20):
    """Back-project SVM / LR weights into original TF-IDF n-grams (Paper Appendix D.1)."""
    if fb.tf is None or fb.tf.svd is None:
        print("Feature builder does not contain TF-IDF SVD.")
        return

    # Extract classifier weights from the pipeline
    clf = model.named_steps.get("linearsvc") or model.named_steps.get("logisticregression")
    scaler = model.named_steps.get("standardscaler")
    if clf is None:
        print("Model is not LinearSVC or LogisticRegression.")
        return

    coef = clf.coef_
    if scaler is not None and hasattr(scaler, "scale_"):
        coef = coef / scaler.scale_

    # SVD dimensionality
    svd_dim = fb.tf.svd.n_components
    svd_weights = coef[0, :svd_dim] if coef.ndim == 2 else coef[:svd_dim]

    res = fb.tf.back_project_weights(svd_weights)
    if res is None:
        return
    vocab, weights = res

    top_ai_idx = np.argsort(weights)[-top_k:][::-1]
    top_human_idx = np.argsort(weights)[:top_k]

    print("\n--- Appendix D.1: Top N-grams Indicating Human vs. AI Code ---")
    print("Top AI-associated tokens (positive weight):")
    for idx in top_ai_idx:
        print(f"  {vocab[idx]:20s} ({weights[idx]:+.4f})")
    print("\nTop Human-associated tokens (negative weight):")
    for idx in top_human_idx:
        print(f"  {vocab[idx]:20s} ({weights[idx]:+.4f})")


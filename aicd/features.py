"""Classical features from Sec. 5.1: whitespace TF-IDF (1-3 grams, 5000 -> SVD 500) and Tree-Sitter AST features."""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

TS_LANG = {"python": "python", "java": "java", "c++": "cpp", "cpp": "cpp", "c": "c", "go": "go", "golang": "go",
           "php": "php", "c#": "c_sharp", "csharp": "c_sharp", "javascript": "javascript", "js": "javascript",
           "rust": "rust"}


class TfidfSvd:
    def __init__(self, max_features=5000, n_components=500):
        self.vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, stop_words=None,
                                   ngram_range=(1, 3), max_features=max_features, dtype=np.float32)
        self.n_components = n_components
        self.svd = None

    def fit_transform(self, texts):
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(min(self.n_components, X.shape[1] - 1), random_state=0)
        return self.svd.fit_transform(X)

    def transform(self, texts):
        return self.svd.transform(self.vec.transform(texts))


_parsers = {}


def _parser(lang):
    from tree_sitter_language_pack import get_parser
    if lang not in _parsers:
        _parsers[lang] = get_parser(lang)
    return _parsers[lang]


def ast_features(code: str, lang: str) -> dict:
    lines = code.split("\n")
    lens = [len(l) for l in lines]
    f = {"n_lines": len(lines), "max_line": max(lens), "avg_line": float(np.mean(lens)),
         "empty_density": sum(1 for l in lines if not l.strip()) / len(lines),
         "alnum_frac": sum(c.isalnum() for c in code) / max(1, len(code)),
         "n_chars": len(code), f"lang={lang}": 1}
    try:
        tree = _parser(TS_LANG.get(lang.lower(), "python")).parse(code.encode("utf8", "ignore"))
    except Exception:
        return f
    depth, n_nodes, errors = 0, 0, 0
    stack = [(tree.root_node, 1)]
    while stack:  # iterative to avoid recursion limits
        node, d = stack.pop()
        n_nodes += 1
        depth = max(depth, d)
        f["t:" + node.type] = f.get("t:" + node.type, 0) + 1
        if node.type == "ERROR":
            errors += 1
        for ch in node.children:
            stack.append((ch, d + 1))
    f.update({"ast_depth": depth, "ast_nodes": n_nodes, "ast_errors": errors})
    return f


class AstFeats:
    def __init__(self):
        self.dv = DictVectorizer(sparse=False)

    def _rows(self, df):
        langs = df["language"] if "language" in df.columns else ["python"] * len(df)
        return [ast_features(c, str(l)) for c, l in zip(df["code"], langs)]

    def fit_transform(self, df):
        return self.dv.fit_transform(self._rows(df)).astype(np.float32)

    def transform(self, df):
        return self.dv.transform(self._rows(df)).astype(np.float32)


class FeatureBuilder:
    def __init__(self, kind):  # tfidf | ast | both
        self.kind = kind
        self.tf = TfidfSvd() if kind in ("tfidf", "both") else None
        self.ast = AstFeats() if kind in ("ast", "both") else None

    def fit_transform(self, df):
        parts = []
        if self.tf: parts.append(self.tf.fit_transform(df["code"].tolist()))
        if self.ast: parts.append(self.ast.fit_transform(df))
        return np.hstack(parts)

    def transform(self, df):
        parts = []
        if self.tf: parts.append(self.tf.transform(df["code"].tolist()))
        if self.ast: parts.append(self.ast.transform(df))
        return np.hstack(parts)

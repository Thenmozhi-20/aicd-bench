"""Classical features from Sec. 5.1: whitespace TF-IDF (1-3 grams, 5000 -> SVD 500) and Tree-Sitter AST features."""
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

TS_LANG = {
    "python": "python", "py": "python",
    "java": "java",
    "c++": "cpp", "cpp": "cpp", "c": "c",
    "go": "go", "golang": "go",
    "php": "php",
    "c#": "c_sharp", "csharp": "c_sharp", "cs": "c_sharp",
    "javascript": "javascript", "js": "javascript",
    "rust": "rust", "rs": "rust"
}

_LANG_HINTS = [
    ("php", re.compile(r"<\?php|\$\w+\s*=")),
    ("rust", re.compile(r"\bfn\s+\w+\s*\(|\blet\s+mut\b|\bimpl\b|\bpub\s+fn\b|println!|\buse\s+std::")),
    ("go", re.compile(r"\bpackage\s+\w+|\bfunc\s+\w*\s*\(|:=|\bfmt\.Print")),
    ("csharp", re.compile(r"\busing\s+System|\bnamespace\s+\w+|\bConsole\.WriteLine|\bpublic\s+static\s+void\s+Main")),
    ("java", re.compile(r"\bpublic\s+class\b|\bimport\s+java\.|System\.out\.print|\bpublic\s+static\s+void\b")),
    ("cpp", re.compile(r"#include\s*<[a-z_]+>\s*(\n.*)?using\s+namespace|std::|\bcout\b|\bcin\b|\bvector<", re.S)),
    ("c", re.compile(r"#include\s*<std\w+\.h>|\bprintf\s*\(|\bscanf\s*\(")),
    ("javascript", re.compile(r"\bfunction\s*\w*\s*\(|=>|\bconsole\.log\b|\bconst\s+\w+\s*=|\blet\s+\w+\s*=")),
    ("python", re.compile(r"\bdef\s+\w+\s*\(|^\s*import\s+\w+|^\s*from\s+\w+\s+import|\bprint\s*\(", re.M)),
]


def guess_language(code: str) -> str:
    """Heuristic identification across the 9 AICD-Bench languages."""
    for name, pat in _LANG_HINTS:
        if pat.search(code):
            return name
    return "python"


class TfidfSvd:
    def __init__(self, max_features=5000, n_components=500):
        self.max_features = max_features
        self.n_components = n_components
        self.vec = TfidfVectorizer(
            token_pattern=r"\S+", lowercase=False, stop_words=None,
            ngram_range=(1, 3), max_features=max_features, dtype=np.float32)
        self.svd = None

    def fit_transform(self, texts):
        X = self.vec.fit_transform(texts)
        k = max(1, min(self.n_components, X.shape[1] - 1))
        self.svd = TruncatedSVD(k, random_state=0)
        return self.svd.fit_transform(X)

    def transform(self, texts):
        return self.svd.transform(self.vec.transform(texts))

    def back_project_weights(self, svd_weights):
        """Back-project hyperplane weights into TF-IDF token space: W_tokens = W_svd * V^T (Appendix D.1)."""
        if self.svd is None or not hasattr(self.svd, "components_"):
            return None
        # svd.components_ shape is (n_components, n_features)
        token_weights = np.dot(svd_weights, self.svd.components_)
        vocab = np.array(self.vec.get_feature_names_out())
        return vocab, token_weights


_parsers = {}


def _parser(lang):
    try:
        from tree_sitter_language_pack import get_parser
        canon = TS_LANG.get(str(lang).lower(), "python")
        if canon not in _parsers:
            _parsers[canon] = get_parser(canon)
        return _parsers[canon]
    except Exception:
        return None


def ast_features(code: str, lang: str = "unknown", max_chars: int = 10000) -> dict:
    code = code[:max_chars]
    lines = code.split("\n")
    lens = [len(l) for l in lines]
    target_lang = lang if lang and lang != "unknown" else guess_language(code)
    f = {
        "n_lines": len(lines),
        "max_line": max(lens) if lens else 0,
        "avg_line": float(np.mean(lens)) if lens else 0.0,
        "empty_density": sum(1 for l in lines if not l.strip()) / max(len(lines), 1),
        "alnum_frac": sum(c.isalnum() for c in code) / max(len(code), 1),
        "indent_tab_frac": sum(l.startswith("\t") for l in lines) / max(len(lines), 1),
        "comment_density": sum(l.lstrip().startswith(("#", "//", "/*", "*")) for l in lines) / max(len(lines), 1),
        "n_chars": len(code),
        f"lang={target_lang}": 1
    }

    parser = _parser(target_lang)
    if parser is None:
        return f

    try:
        tree = parser.parse(code.encode("utf8", "ignore"))
    except Exception:
        return f

    depth, n_nodes, errors = 0, 0, 0
    construct_counts = {"if": 0, "loop": 0, "func": 0}
    stack = [(tree.root_node, 1)]
    while stack:  # iterative walk
        node, d = stack.pop()
        n_nodes += 1
        depth = max(depth, d)
        ntype = node.type
        f["t:" + ntype] = f.get("t:" + ntype, 0) + 1
        
        # Track construct types per paper Section 5.1
        if "if" in ntype:
            construct_counts["if"] += 1
        if any(lp in ntype for lp in ("for", "while", "loop")):
            construct_counts["loop"] += 1
        if any(fn in ntype for fn in ("function", "method", "fn_def")):
            construct_counts["func"] += 1

        if ntype == "ERROR":
            errors += 1
        for ch in node.children:
            stack.append((ch, d + 1))

    f.update({
        "ast_depth": depth,
        "ast_nodes": n_nodes,
        "ast_errors": errors,
        "n_if": construct_counts["if"],
        "n_loop": construct_counts["loop"],
        "n_func": construct_counts["func"],
    })
    # Node density normalized by total nodes
    for k in list(f.keys()):
        if k.startswith("t:"):
            f["norm_" + k] = f[k] / max(n_nodes, 1)

    return f


class AstFeats:
    def __init__(self):
        self.dv = DictVectorizer(sparse=False)

    def _rows(self, df):
        from tqdm import tqdm
        if "language" in df.columns:
            langs = [str(l) for l in df["language"]]
        else:
            langs = [guess_language(c) for c in df["code"]]
        return [ast_features(c, l) for c, l in tqdm(zip(df["code"], langs), total=len(df), desc="Extracting AST features", unit="code", mininterval=0.5)]

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
        return np.hstack(parts) if len(parts) > 1 else parts[0]

    def transform(self, df):
        parts = []
        if self.tf: parts.append(self.tf.transform(df["code"].tolist()))
        if self.ast: parts.append(self.ast.transform(df))
        return np.hstack(parts) if len(parts) > 1 else parts[0]


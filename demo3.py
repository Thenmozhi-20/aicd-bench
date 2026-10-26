import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from aicd.features import TfidfSvd

NAMES = {0: "human", 1: "AI", 2: "hybrid (edited by AI)", 3: "adversarial"}  # assumed label order, not verified
code = open(sys.argv[1], encoding="utf8").read()
df = pd.read_parquet("data/T3_train.parquet").sample(5000, random_state=0)
tf = TfidfSvd()
X = tf.fit_transform(df["code"].tolist())
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X, df["label"])
proba = clf.predict_proba(tf.transform([code]))[0]
for c, p in sorted(zip(clf.classes_, proba), key=lambda x: -x[1]):
    print(f"{NAMES[int(c)]:24s} {p:.0%}")

import joblib
import pandas as pd
from train_model import build_features

bundle = joblib.load("rf_plasticizer_model.joblib")
model = bundle["model"]
domain_vocab = bundle["domain_vocab"]

ids, X = build_features(
    "KSSKSLAB04_esterases.faa",
    "KSSK_esterase.pfam.domtblout",
    domain_vocab
)

scores = model.predict_proba(X)[:,1]

df = pd.DataFrame({
    "protein_id": ids,
    "plasticizer_score": scores
})

df = df.sort_values("plasticizer_score", ascending=False)
df.to_csv("KSSKSLAB04_plasticizer_ranked.csv", index=False)

print(df.head(20))

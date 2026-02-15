import pandas as pd
import os
import matplotlib.pyplot as plt
os.chdir(r"C:\Users\Anuraj\Desktop\khushboo_backup\ai_ml_ksskslab")
abc= pd.read_csv("feature_importance_top30.csv")
Top_N= 15
abc_top=abc.head(Top_N).copy()
abc_top=abc_top.iloc[::-1]
plt.figure(figsize=(8, 6))
plt.barh(abc_top["feature"], abc_top["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title(f"Top {Top_N} Predictive Features (Random Forest)")
plt.tight_layout()
plt.savefig("Features_importance", dpi=300, bbox_inches="tight")
plt.show()

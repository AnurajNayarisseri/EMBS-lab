import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load kinome data
df = pd.read_csv("kinome_profile.csv")

# Sort by CDK2-relative binding
df = df.sort_values("Relative_to_CDK2", ascending=False)

# Set kinase names as index
df = df.set_index("Kinase")

# Use only relative binding for heatmap
data = df[["Relative_to_CDK2"]]

plt.figure(figsize=(4, 14))
sns.heatmap(
    data,
    cmap="coolwarm",
    linewidths=0.3,
    cbar_kws={"label": "Relative binding vs CDK2"},
    yticklabels=True
)

plt.title("Kinome-wide selectivity of CDKCRC-EMBS")
plt.tight_layout()
plt.savefig("kinome_heatmap.png", dpi=300)
plt.close()

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from train_model import build_features, build_domain_vocab

# Load data
POS_FASTA = "pos.faa"
NEG_FASTA = "neg_clean.faa"
POS_DOM = "pos.pfam.domtblout"
NEG_DOM = "neg_clean.pfam.domtblout"

print("Building domain vocabulary...")
domain_vocab = build_domain_vocab(POS_DOM, NEG_DOM)

print("Extracting features...")
_, X_pos = build_features(POS_FASTA, POS_DOM, domain_vocab)
_, X_neg = build_features(NEG_FASTA, NEG_DOM, domain_vocab)

X = np.vstack([X_pos, X_neg])
y = np.array([1]*len(X_pos) + [0]*len(X_neg))

# Scale features (important for neural networks)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Build neural network
model = keras.Sequential([
    keras.layers.Dense(256, activation="relu", input_shape=(X.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["AUC"]
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
probs = model.predict(X_test).ravel()

print("\n=== Keras Model Performance ===")
print("ROC-AUC:", roc_auc_score(y_test, probs))
print("PR-AUC :", average_precision_score(y_test, probs))

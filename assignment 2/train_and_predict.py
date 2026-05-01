"""
Kaggle Multiclass Classification — Irrigation Need Prediction
=============================================================
Models trained:
  - Decision Tree        (submitted)
  - Naive Bayes
  - Logistic Regression
  - K-Nearest Neighbours
  - K-Means (as classifier)
  - Random Forest        (best CV score)

Cross-validation:
  - 5-Fold Stratified K-Fold on Random Forest
  - LOOCV discussed but impractical at 630k rows
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TRAIN_PATH = 'train__1_.csv'
TEST_PATH  = 'test.csv'
SUB_PATH   = 'sample_submission.csv'
OUTPUT_PATH = 'submission.csv'
TARGET = 'Irrigation_Need'

CAT_COLS = [
    'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
    'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region'
]
NUM_COLS = [
    'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
    'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
    'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm'
]
FEATURES = CAT_COLS + NUM_COLS

# ─── LOAD ─────────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sub   = pd.read_csv(SUB_PATH)
print(f"  Train: {train.shape}  |  Test: {test.shape}")
print(f"  Missing values: {train.isnull().sum().sum()}")
print(f"  Class distribution:\n{train[TARGET].value_counts()}\n")

# ─── ENCODE ──────────────────────────────────────────────────────────────────
print("Encoding features...")
for col in CAT_COLS:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

le_y = LabelEncoder()
train[TARGET] = le_y.fit_transform(train[TARGET])
CLASSES = le_y.classes_
print(f"  Classes: {CLASSES}\n")

# ─── SPLIT ───────────────────────────────────────────────────────────────────
X = train[FEATURES].values
y = train[TARGET].values
X_test_full = test[FEATURES].values

# 60k stratified sample for evaluation (full data used for submission)
X_samp, _, y_samp, _ = train_test_split(
    X, y, train_size=60_000, stratify=y, random_state=42
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_samp, y_samp, test_size=0.2, stratify=y_samp, random_state=42
)

# Scale for distance/linear models
sc = StandardScaler()
X_tr_s  = sc.fit_transform(X_tr)
X_val_s = sc.transform(X_val)

print(f"Train sample: {X_tr.shape}  |  Val: {X_val.shape}\n")
print("=" * 65)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
results = {}

def evaluate(name, model, X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    acc = accuracy_score(y_valid, preds)
    f1  = f1_score(y_valid, preds, average='weighted')
    cm  = confusion_matrix(y_valid, preds)
    results[name] = {'model': model, 'acc': acc, 'f1': f1, 'cm': cm}
    print(f"[{name}]")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 (wtd) : {f1:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    print(f"  Classification report:\n{classification_report(y_valid, preds, target_names=CLASSES)}")
    return model

# ─── 1. DECISION TREE ────────────────────────────────────────────────────────
evaluate(
    'Decision Tree',
    DecisionTreeClassifier(max_depth=15, random_state=42),
    X_tr, X_val, y_tr, y_val
)

# ─── 2. NAIVE BAYES ──────────────────────────────────────────────────────────
evaluate(
    'Naive Bayes',
    GaussianNB(),
    X_tr_s, X_val_s, y_tr, y_val
)

# ─── 3. LOGISTIC REGRESSION ──────────────────────────────────────────────────
evaluate(
    'Logistic Regression',
    LogisticRegression(max_iter=300, C=1.0, random_state=42),
    X_tr_s, X_val_s, y_tr, y_val
)

# ─── 4. K-NEAREST NEIGHBOURS ─────────────────────────────────────────────────
evaluate(
    'KNN (k=11)',
    KNeighborsClassifier(n_neighbors=11, n_jobs=-1),
    X_tr_s, X_val_s, y_tr, y_val
)

# ─── 5. K-MEANS AS CLASSIFIER ────────────────────────────────────────────────
print("[K-Means Classifier]")
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X_tr_s)
cluster_tr  = km.predict(X_tr_s)
cluster_val = km.predict(X_val_s)

# Map each cluster to its majority class
cluster_map = {}
for c in range(3):
    mask = cluster_tr == c
    cluster_map[c] = int(np.bincount(y_tr[mask]).argmax())

preds_km = np.array([cluster_map[c] for c in cluster_val])
acc_km = accuracy_score(y_val, preds_km)
cm_km  = confusion_matrix(y_val, preds_km)
results['KMeans'] = {'acc': acc_km, 'cm': cm_km}
print(f"  Accuracy : {acc_km:.4f}")
print(f"  Confusion matrix:\n{cm_km}")
print(f"  Note: K-Means clusters ≠ class boundaries — unsupervised mismatch.\n")

# ─── 6. RANDOM FOREST ────────────────────────────────────────────────────────
evaluate(
    'Random Forest',
    RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42),
    X_tr, X_val, y_tr, y_val
)

# ─── SUMMARY TABLE ───────────────────────────────────────────────────────────
print("=" * 65)
print("MODEL SUMMARY (sorted by accuracy)")
print("=" * 65)
for name, m in sorted(results.items(), key=lambda x: x[1]['acc'], reverse=True):
    if 'f1' in m:
        print(f"  {name:<25}  Acc={m['acc']:.4f}  F1={m['f1']:.4f}")
    else:
        print(f"  {name:<25}  Acc={m['acc']:.4f}")

# ─── 5-FOLD CV ON RANDOM FOREST ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("5-FOLD STRATIFIED K-FOLD CV — Random Forest (60k sample)")
print("=" * 65)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_samp, y_samp)):
    rf = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_samp[tr_idx], y_samp[tr_idx])
    p = rf.predict(X_samp[val_idx])
    fa = accuracy_score(y_samp[val_idx], p)
    fold_accs.append(fa)
    print(f"  Fold {fold+1}: {fa:.4f}")
print(f"  CV Mean = {np.mean(fold_accs):.4f}  |  Std = {np.std(fold_accs):.4f}")

# ─── GENERATE SUBMISSION ─────────────────────────────────────────────────────
# Best submission model: Decision Tree trained on FULL 630k data
print("\nGenerating submission with Decision Tree (full training data)...")
dt_final = DecisionTreeClassifier(max_depth=20, random_state=42)
dt_final.fit(X, y)
test_preds = dt_final.predict(X_test_full)
sub[TARGET] = le_y.inverse_transform(test_preds)
sub.to_csv(OUTPUT_PATH, index=False)
print(f"Submission saved to {OUTPUT_PATH}")
print(sub[TARGET].value_counts().to_dict())

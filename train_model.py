# ============================================================
# train_model.py
# Trains XGBoost + RF + Extra Trees on NSL-KDD
# Saves model to rf_model.pkl
# Run this ONCE
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

BASE        = r'D:\GT_Research_paper'
TRAIN_FILE  = os.path.join(BASE, 'KDDTrain+.txt')
TEST_FILE   = os.path.join(BASE, 'KDDTest+.txt')
MODEL_FILE  = os.path.join(BASE, 'rf_model.pkl')
EXCEL_FILE  = os.path.join(BASE, 'ml_results.xlsx')
LOG_FILE    = os.path.join(BASE, 'training_log.txt')

COLUMNS = [
    'duration','protocol_type','service','flag',
    'src_bytes','dst_bytes','land','wrong_fragment',
    'urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login',
    'count','srv_count',
    'serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'attack_type','difficulty'
]

CATEGORICAL  = ['protocol_type','service','flag']
DROP_COLS    = ['attack_type','difficulty','category']
NUMERIC      = [c for c in COLUMNS if c not in CATEGORICAL + DROP_COLS]
FEATURE_COLS = CATEGORICAL + NUMERIC

ATTACK_MAP = {
    'normal':'Normal',
    'back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS',
    'smurf':'DoS','teardrop':'DoS','apache2':'DoS','udpstorm':'DoS',
    'processtable':'DoS','worm':'DoS','mailbomb':'DoS',
    'ipsweep':'Probe','nmap':'Probe','portsweep':'Probe',
    'satan':'Probe','mscan':'Probe','saint':'Probe',
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L',
    'multihop':'R2L','phf':'R2L','spy':'R2L','warezclient':'R2L',
    'warezmaster':'R2L','xlock':'R2L','xsnoop':'R2L',
    'snmpguess':'R2L','snmpgetattack':'R2L','httptunnel':'R2L',
    'sendmail':'R2L','named':'R2L',
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R',
    'rootkit':'U2R','sqlattack':'U2R','xterm':'U2R','ps':'U2R'
}

CLASS_ORDER = ['DoS','Normal','Probe','R2L','U2R']

log_lines = []
def log(msg=''):
    print(msg)
    log_lines.append(str(msg))

def section(title):
    log(); log('='*60); log(title); log('='*60)

# ============================================================
# STEP 1 — LOAD AND COMBINE BOTH FILES
# ============================================================
section('STEP 1 — Loading data')

tr = pd.read_csv(TRAIN_FILE, header=None, names=COLUMNS)
te = pd.read_csv(TEST_FILE,  header=None, names=COLUMNS)

log(f'KDDTrain+ : {len(tr):,}')
log(f'KDDTest+  : {len(te):,}')

df = pd.concat([tr, te], ignore_index=True)
df['category'] = df['attack_type'].map(ATTACK_MAP).fillna('Other')
df = df[df['category'] != 'Other'].reset_index(drop=True)

log(f'Combined  : {len(df):,}')
log('\nDistribution:')
for cat, cnt in df['category'].value_counts().items():
    pct = cnt/len(df)*100
    bar = '█' * int(pct/2)
    log(f'  {cat:8s}: {cnt:6,} ({pct:4.1f}%) {bar}')

# ============================================================
# STEP 2 — ENCODE CATEGORICAL
# ============================================================
section('STEP 2 — Encoding categorical columns')

le_store = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    le.fit(df[col])
    df[col]       = le.transform(df[col])
    le_store[col] = le
    log(f'  {col}: {len(le.classes_)} values encoded')

# ============================================================
# STEP 3 — ENCODE LABELS
# ============================================================
section('STEP 3 — Encoding labels')

label_enc = LabelEncoder()
label_enc.fit(CLASS_ORDER)
df['label'] = label_enc.transform(df['category'])

log('Label mapping:')
for i, cls in enumerate(label_enc.classes_):
    log(f'  {cls:8s} → {i}')

# ============================================================
# STEP 4 — STRATIFIED SPLIT
# ============================================================
section('STEP 4 — Stratified 80/20 split')

X_all = df[FEATURE_COLS].copy()
y_all = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20,
    random_state=42, stratify=y_all)

log(f'Train : {len(X_train):,}  Test : {len(X_test):,}')

# ============================================================
# STEP 5 — NORMALIZE
# ============================================================
section('STEP 5 — MinMax normalization')

scaler = MinMaxScaler()
X_train[NUMERIC] = scaler.fit_transform(X_train[NUMERIC])
X_test[NUMERIC]  = scaler.transform(X_test[NUMERIC])
log('All numeric features scaled to [0, 1]')

# ============================================================
# STEP 6 — SMOTE
# ============================================================
section('STEP 6 — SMOTE oversampling')

log('Before SMOTE:')
u, c = np.unique(y_train, return_counts=True)
for uu, cc in zip(u, c):
    log(f'  {CLASS_ORDER[uu]:8s}: {cc:,}')

n_normal = int((y_train == 1).sum())
smote_strategy = {
    0: n_normal, 1: n_normal, 2: n_normal,
    3: min(n_normal, 15000),
    4: min(n_normal, 10000)
}

smote    = SMOTE(sampling_strategy=smote_strategy,
                  random_state=42, k_neighbors=5)
X_sm, y_sm = smote.fit_resample(X_train.values, y_train)
X_sm = pd.DataFrame(X_sm, columns=FEATURE_COLS)

log(f'\nAfter SMOTE: {len(X_sm):,}')
u2, c2 = np.unique(y_sm, return_counts=True)
for uu, cc in zip(u2, c2):
    log(f'  {CLASS_ORDER[uu]:8s}: {cc:,}')

# ============================================================
# STEP 7 — TRAIN THREE MODELS
# ============================================================
section('STEP 7 — Training three models (8-15 mins)')

log('Training XGBoost...')
xgb = XGBClassifier(
    n_estimators=500, max_depth=10, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', random_state=42,
    n_jobs=-1, verbosity=0)
xgb.fit(X_sm.values, y_sm)
log('XGBoost done.')

log('Training Random Forest...')
rf = RandomForestClassifier(
    n_estimators=300, max_depth=None,
    max_features='sqrt', random_state=42, n_jobs=-1)
rf.fit(X_sm.values, y_sm)
log('Random Forest done.')

log('Training Extra Trees...')
et = ExtraTreesClassifier(
    n_estimators=300, max_depth=None,
    max_features='sqrt', random_state=42, n_jobs=-1)
et.fit(X_sm.values, y_sm)
log('Extra Trees done.')

# ============================================================
# STEP 8 — ENSEMBLE PREDICTIONS
# ============================================================
section('STEP 8 — Ensemble predictions')

Xte       = X_test.values
p_xgb     = xgb.predict_proba(Xte)
p_rf      = rf.predict_proba(Xte)
p_et      = et.predict_proba(Xte)
p_ens     = 0.50*p_xgb + 0.25*p_rf + 0.25*p_et
y_pred    = np.argmax(p_ens, axis=1)

acc_xgb = accuracy_score(y_test, np.argmax(p_xgb,axis=1))*100
acc_rf  = accuracy_score(y_test, np.argmax(p_rf, axis=1))*100
acc_et  = accuracy_score(y_test, np.argmax(p_et, axis=1))*100
acc_ens = accuracy_score(y_test, y_pred)*100

log(f'XGBoost  : {acc_xgb:.4f}%')
log(f'RF       : {acc_rf:.4f}%')
log(f'ET       : {acc_et:.4f}%')
log(f'Ensemble : {acc_ens:.4f}%  ← best')

# ============================================================
# STEP 9 — METRICS
# ============================================================
section('STEP 9 — Final metrics')

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average=None,
                        labels=range(5), zero_division=0)
rec  = recall_score(y_test, y_pred, average=None,
                     labels=range(5), zero_division=0)
f1s  = f1_score(y_test, y_pred, average=None,
                 labels=range(5), zero_division=0)

log(f'Overall Accuracy : {acc*100:.4f}%')
log()
log(f'{"Class":8s}  {"Precision":>10s}  {"Recall":>10s}  {"F1":>10s}  {"N":>7s}')
log('-'*55)
for i, cls in enumerate(CLASS_ORDER):
    n = int((y_test==i).sum())
    log(f'{cls:8s}  {prec[i]*100:>9.2f}%  {rec[i]*100:>9.2f}%  {f1s[i]*100:>9.2f}%  {n:>7,}')
log()
log(f'Macro F1 : {f1s.mean()*100:.2f}%')
log()
log(classification_report(y_test, y_pred,
    target_names=CLASS_ORDER, digits=4, zero_division=0))

# ============================================================
# STEP 10 — OVERALL DETECTION PROBABILITIES
# These are averages across all test records
# Used for general paper results comparison
# ============================================================
section('STEP 10 — Overall detection probabilities')

det_probs  = {}
attack_cls = [c for c in CLASS_ORDER if c != 'Normal']

log('Average detection probability per attack type:')
log('(averaged across all test records of that type)')
log()
for cls in attack_cls:
    idx  = list(label_enc.classes_).index(cls)
    mask = y_test == idx
    n    = int(mask.sum())
    prob = float(np.mean(p_ens[mask, idx])) if n > 0 else 0.5
    det_probs[cls] = round(prob, 6)
    bar = '█' * int(prob*25)
    log(f'  {cls:6s}: {prob:.4f}  (from {n:,} records)  {bar}')

# ============================================================
# STEP 11 — CONFUSION MATRIX GRAPH
# ============================================================
section('STEP 11 — Saving graphs')

cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=axes[0])
axes[0].set_title(f'Confusion Matrix — Counts\nAccuracy: {acc*100:.2f}%')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

cm_pct = cm.astype(float)
for i in range(5):
    if cm[i].sum() > 0:
        cm_pct[i] = cm[i]/cm[i].sum()*100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=axes[1])
axes[1].set_title('Confusion Matrix — Percentages')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')
plt.suptitle('NSL-KDD XGBoost+RF+ET Ensemble', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
log('Saved: confusion_matrix.png')

# Feature importance
fi_df = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=fi_df.head(20), x='importance', y='feature', palette='viridis', ax=ax)
ax.set_title('Top 20 Feature Importances — XGBoost')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
log('Saved: feature_importance.png')

# ============================================================
# STEP 12 — SAVE EXCEL
# ============================================================
section('STEP 12 — Saving Excel')

with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as w:
    rows = [{'Metric':'Overall Accuracy (%)', 'Value': round(acc*100,4)}]
    for i, cls in enumerate(CLASS_ORDER):
        rows.append({
            'Metric': cls,
            'Precision': round(prec[i]*100,2),
            'Recall':    round(rec[i]*100,2),
            'F1-Score':  round(f1s[i]*100,2),
            'Support':   int((y_test==i).sum())
        })
    pd.DataFrame(rows).to_excel(w, sheet_name='Metrics', index=False)

    dp = [{'Attack': cls, 'Detection Prob': det_probs[cls],
           'Attack Cost': {'DoS':2,'Probe':1,'R2L':3,'U2R':4}[cls]}
          for cls in attack_cls]
    pd.DataFrame(dp).to_excel(w, sheet_name='Detection Probs', index=False)
    fi_df.to_excel(w, sheet_name='Feature Importance', index=False)

log('Saved: ml_results.xlsx')

# ============================================================
# STEP 13 — SAVE MODEL BUNDLE
# ============================================================
section('STEP 13 — Saving model bundle')

bundle = {
    'xgb':       xgb,
    'rf':        rf,
    'et':        et,
    'scaler':    scaler,
    'label_enc': label_enc,
    'det_probs': det_probs,
    'classes':   CLASS_ORDER,
    'features':  FEATURE_COLS,
    'weights':   (0.50, 0.25, 0.25),
    'accuracy':  round(acc*100, 4)
}

with open(MODEL_FILE, 'wb') as f:
    pickle.dump(bundle, f)

log(f'Saved: rf_model.pkl')

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))

# ============================================================
# FINAL SUMMARY
# ============================================================
section('TRAINING COMPLETE')
log(f'Accuracy      : {acc*100:.4f}%')
log(f'Train records : {len(X_sm):,} (after SMOTE)')
log(f'Test records  : {len(X_test):,}')
log()
log('Detection probabilities (overall averages):')
for cls in attack_cls:
    p   = det_probs[cls]
    bar = '█' * int(p*25)
    log(f'  {cls:6s}: {p:.4f}  {bar}')
log()
log('NEXT: Run game_theory.py')
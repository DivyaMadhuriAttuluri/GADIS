# ============================================================
# game_theory.py
# For ONE connection record:
# 1. ML model predicts attack probabilities
# 2. Vulnerability weights applied per node
# 3. Auto-negative check
# 4. Payoff calculated for all 20 combinations
# 5. Stackelberg solver finds Nash Equilibrium r*
# 6. Deploy security allocation
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize

BASE       = r'D:\GT_Research_paper'
MODEL_FILE = os.path.join(BASE, 'rf_model.pkl')
TRAIN_FILE = os.path.join(BASE, 'KDDTrain+.txt')
TEST_FILE  = os.path.join(BASE, 'KDDTest+.txt')

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

CLASS_ORDER  = ['DoS','Normal','Probe','R2L','U2R']
ATTACK_TYPES = ['DoS','Probe','R2L','U2R']

# ============================================================
# 5 NODES
# ============================================================

NODES = ['Database','Admin','Email','File','Web']
N     = 5

NODE_VALUES = {
    'Database': 10,
    'Admin'   :  8,
    'Email'   :  6,
    'File'    :  4,
    'Web'     :  3
}

ATTACK_COSTS = {
    'DoS'  : 2,
    'Probe': 1,
    'R2L'  : 3,
    'U2R'  : 4
}

# ============================================================
# VULNERABILITY WEIGHTS
# How susceptible each node is to each attack type
# Non-zero always — even unlikely combos have small chance
#
#              DoS    Probe   R2L    U2R
# Database:    0.3    0.8     1.0    0.7
# Admin:       0.1    0.6     0.4    1.0
# Email:       0.4    0.7     1.0    0.5
# File:        0.2    0.6     0.7    0.4
# Web:         1.0    1.0     0.1    0.05
#
# DoS  → primarily targets public servers (Web highest)
# Probe→ scans everything (Web and Database high)
# R2L  → targets login servers (Database, Email highest)
# U2R  → targets privilege servers (Admin highest)
# ============================================================

VULN = {
    'Database': {'DoS':0.30, 'Probe':0.80, 'R2L':1.00, 'U2R':0.70},
    'Admin'   : {'DoS':0.10, 'Probe':0.60, 'R2L':0.40, 'U2R':1.00},
    'Email'   : {'DoS':0.40, 'Probe':0.70, 'R2L':1.00, 'U2R':0.50},
    'File'    : {'DoS':0.20, 'Probe':0.60, 'R2L':0.70, 'U2R':0.40},
    'Web'     : {'DoS':1.00, 'Probe':1.00, 'R2L':0.10, 'U2R':0.05},
}

MIN_R = 0.01  # every node gets at least 1% monitoring

# ============================================================
# CHANGE THIS TO TEST DIFFERENT RECORDS
#
#   0      → DoS attack
#   60000  → Normal traffic
#   120000 → Probe attack
#   135000 → R2L attack
#   148400 → U2R attack
# ============================================================

TEST_INDEX = 17

# ============================================================
# CORE FUNCTIONS
# ============================================================

def effective_prob(node, atk, ml_prob):
    """
    Effective probability = ML probability × vulnerability weight
    Accounts for how susceptible this node is to this attack
    """
    return ml_prob * VULN[node][atk]

def payoff(node, atk, ri, ml_probs):
    """
    Attacker profit formula:
    U_A = V(i) × effective_prob(node,t) × (1 - ri) - C(t)

    V(i)             = server value if compromised
    effective_prob   = ML_prob × vulnerability_weight
    (1 - ri)         = attacker success factor
                       more resource → less success for Ravi
    C(t)             = attack cost
    """
    V    = NODE_VALUES[node]
    C    = ATTACK_COSTS[atk]
    ep   = effective_prob(node, atk, ml_probs[atk])
    return V * ep * (1 - ri) - C

def is_auto_negative(atk, ml_probs):
    """
    Check if this attack is already negative on ALL nodes
    even with zero resource.

    Condition: max possible prize < cost
    max_prize = max_V × ML_prob × max_vuln_weight

    If true → no resource needed → skip in solver
    """
    p        = ml_probs[atk]
    C        = ATTACK_COSTS[atk]
    max_V    = max(NODE_VALUES.values())
    max_vuln = max(VULN[node][atk] for node in NODES)
    max_prize = max_V * p * max_vuln
    return max_prize < C

def max_payoff_fn(r_vec, ml_probs, active_attacks):
    """
    Ravi's best possible payoff across all
    active (node, attack) combinations.
    This is what the defender minimises.
    """
    best = -9999
    for i, node in enumerate(NODES):
        for atk in active_attacks:
            p = payoff(node, atk, r_vec[i], ml_probs)
            if p > best:
                best = p
    return best

def best_attack_fn(r_vec, ml_probs, active_attacks):
    """Which node+attack gives Ravi maximum profit?"""
    best_p = -9999
    best_n = None
    best_a = None
    for i, node in enumerate(NODES):
        for atk in active_attacks:
            p = payoff(node, atk, r_vec[i], ml_probs)
            if p > best_p:
                best_p = p
                best_n = node
                best_a = atk
    return best_n, best_a, best_p

def solve_stackelberg(ml_probs, active_attacks):
    """
    Stackelberg solver — backward induction.
    Defender finds r* that minimises attacker best payoff.
    Uses smart starting point biased toward
    nodes vulnerable to active attacks.
    """
    # Smart starting allocation
    r0 = np.ones(N) * MIN_R
    for i, node in enumerate(NODES):
        for atk in active_attacks:
            r0[i] += NODE_VALUES[node] * ml_probs[atk] * VULN[node][atk]
    r0 = r0 / r0.sum()

    result = minimize(
        fun=lambda r: max_payoff_fn(r, ml_probs, active_attacks),
        x0=r0,
        method='SLSQP',
        bounds=[(MIN_R, 0.95)] * N,
        constraints=[{'type':'eq', 'fun': lambda r: np.sum(r) - 1.0}],
        options={'ftol': 1e-15, 'maxiter': 100000}
    )
    return result.x, result.success

# ============================================================
# LOAD MODEL
# ============================================================

print()
print('='*65)
print('Loading trained model...')
print('='*65)

with open(MODEL_FILE, 'rb') as f:
    bundle = pickle.load(f)

xgb     = bundle['xgb']
rf      = bundle['rf']
et      = bundle['et']
scaler  = bundle['scaler']
weights = bundle['weights']

print(f'Model loaded. Training accuracy: {bundle["accuracy"]}%')

# ============================================================
# LOAD DATASET AND PICK ONE RECORD
# ============================================================

tr = pd.read_csv(TRAIN_FILE, header=None, names=COLUMNS)
te = pd.read_csv(TEST_FILE,  header=None, names=COLUMNS)
df = pd.concat([tr, te], ignore_index=True)
df['category'] = df['attack_type'].map(ATTACK_MAP).fillna('Other')
df = df[df['category'] != 'Other'].reset_index(drop=True)

for col in CATEGORICAL:
    le = LabelEncoder()
    le.fit(df[col])
    df[col] = le.transform(df[col])

true_label = df.loc[TEST_INDEX, 'category']
record     = df.iloc[[TEST_INDEX]][FEATURE_COLS].copy()
record[NUMERIC] = scaler.transform(record[NUMERIC])

print(f'Dataset loaded: {len(df):,} records')
print(f'Testing record index: {TEST_INDEX}')

# ============================================================
# STAGE 1 — ML MODEL PREDICTION
# ============================================================

print()
print('='*65)
print('STAGE 1 — ML MODEL PREDICTION')
print('What type of attack is this connection?')
print('='*65)

X     = record.values
p_xgb = xgb.predict_proba(X)[0]
p_rf  = rf.predict_proba(X)[0]
p_et  = et.predict_proba(X)[0]

# Weighted ensemble probability for THIS specific record
p_record = weights[0]*p_xgb + weights[1]*p_rf + weights[2]*p_et

pred_idx   = int(np.argmax(p_record))
pred_label = CLASS_ORDER[pred_idx]
confidence = p_record[pred_idx]

# Extract attack type probabilities
ml_probs = {cls: float(p_record[CLASS_ORDER.index(cls)])
             for cls in ATTACK_TYPES}

print()
print(f'Record index : {TEST_INDEX}')
print(f'True label   : {true_label}')
print()
print('ML Output — how strongly this connection')
print('shows signs of each attack type:')
print()
print(f'  {"Category":8s}  {"Probability":>12s}  Visual')
print('  ' + '-'*55)

for i, cls in enumerate(CLASS_ORDER):
    p   = p_record[i]
    bar = '█' * int(p*40)
    mrk = ' ← PREDICTED' if i == pred_idx else ''
    print(f'  {cls:8s}  {p*100:>11.4f}%  {bar}{mrk}')

status = 'CORRECT' if pred_label == true_label else 'INCORRECT'
print()
print(f'Predicted : {pred_label} ({confidence*100:.2f}%) — {status}')

if pred_label == 'Normal':
    print()
    print('Normal traffic detected. No attack.')
    print('No security reallocation needed.')
    exit()

# ============================================================
# STAGE 2 — AUTO NEGATIVE CHECK
# ============================================================

print()
print('='*65)
print('STAGE 2 — AUTO NEGATIVE CHECK')
print('Which attacks are already safe even')
print('with zero resource? (max_prize < cost)')
print('='*65)
print()
print(f'{"Attack":8s}  {"ML Prob":>8s}  {"Max Prize":>10s}  {"Cost":>6s}  {"Result"}')
print('-'*60)

auto_neg    = {}
active_atks = []
max_V       = max(NODE_VALUES.values())

for atk in ATTACK_TYPES:
    p         = ml_probs[atk]
    C         = ATTACK_COSTS[atk]
    max_vuln  = max(VULN[nd][atk] for nd in NODES)
    max_prize = max_V * p * max_vuln
    is_auto   = max_prize < C
    auto_neg[atk] = is_auto
    if not is_auto:
        active_atks.append(atk)
    tag = 'AUTO NEGATIVE — safe everywhere' if is_auto else 'ACTIVE THREAT — needs resource'
    print(f'{atk:8s}  {p*100:>7.2f}%  {max_prize:>10.4f}  {C:>6d}  {tag}')

print()
print(f'Auto-negative : {[a for a in ATTACK_TYPES if auto_neg[a]]}')
print(f'Active threats: {active_atks}')

if not active_atks:
    print()
    print('ALL ATTACKS AUTO NEGATIVE.')
    print('Equal allocation is sufficient.')
    print('No game theory needed.')
    exit()

# ============================================================
# STAGE 3 — EFFECTIVE PROBABILITY PER NODE
# ============================================================

print()
print('='*65)
print('STAGE 3 — EFFECTIVE PROBABILITY PER NODE')
print('= ML_prob × vulnerability_weight')
print('Accounts for how susceptible each')
print('node is to each attack type')
print('='*65)
print()

for atk in active_atks:
    print(f'{atk} (ML prob = {ml_probs[atk]*100:.2f}%):')
    print(f'  {"Node":10s}  {"Vuln Wt":>8s}  {"Eff Prob":>10s}  {"Meaning"}')
    print('  ' + '-'*60)
    for node in NODES:
        vw  = VULN[node][atk]
        ep  = ml_probs[atk] * vw
        if vw >= 0.80:
            meaning = 'Primary target'
        elif vw >= 0.50:
            meaning = 'Secondary target'
        elif vw >= 0.20:
            meaning = 'Possible target'
        else:
            meaning = 'Unlikely but non-zero'
        print(f'  {node:10s}  {vw:>8.2f}  {ep:>10.4f}  {meaning}')
    print()

# ============================================================
# STAGE 4 — PAYOFF FOR ALL 20 COMBINATIONS
# ============================================================

print()
print('='*65)
print('STAGE 4 — PAYOFF FOR ALL 20 COMBINATIONS')
print('U_A = V × ML_prob × vuln_weight × (1-ri) - C')
print('Positive = Ravi profits = DANGER')
print('Negative = Ravi loses   = SAFE')
print('='*65)
print()

r_equal = 1.0 / N
print(f'At equal allocation ({r_equal*100:.0f}% each node):')
print()
print(f'  {"Node":10s}  {"V":>3s}  ' +
      '  '.join(f'{a:>10s}' for a in ATTACK_TYPES))
print('  ' + '-'*70)

has_positive = False
danger_list  = []

for node in NODES:
    row = f'  {node:10s}  {NODE_VALUES[node]:>3d}  '
    for atk in ATTACK_TYPES:
        if auto_neg[atk]:
            p = payoff(node, atk, r_equal, ml_probs)
            row += f'{"auto("+str(round(p,2))+")":>12s}  '
        else:
            p = payoff(node, atk, r_equal, ml_probs)
            if p > 0:
                row += f'{p:>+12.3f}  '
                has_positive = True
                danger_list.append((node, atk, p))
            else:
                row += f'{p:>12.3f}  '
    print(row)

print()
if danger_list:
    print(f'DANGER ZONES ({len(danger_list)} combinations):')
    for node, atk, p in sorted(danger_list, key=lambda x: x[2], reverse=True):
        print(f'  {node:10s} + {atk:6s} = +{p:.4f}  ← Ravi profits here')
else:
    print('All payoffs already negative at equal allocation.')

# ============================================================
# STAGE 5 — MINIMUM RESOURCE NEEDED
# ============================================================

print()
print('='*65)
print('STAGE 5 — MINIMUM RESOURCE PER NODE')
print('ri > 1 - C/(V × eff_prob)')
print('='*65)
print()

total_min = 0
for atk in active_atks:
    print(f'{atk}:')
    for node in NODES:
        V   = NODE_VALUES[node]
        ep  = effective_prob(node, atk, ml_probs[atk])
        C   = ATTACK_COSTS[atk]
        p0  = payoff(node, atk, 0, ml_probs)
        if V * ep > C:
            ri_needed = 1 - C/(V*ep)
        else:
            ri_needed = 0.0
        total_min += max(ri_needed, MIN_R)
        status = 'already safe' if p0 <= 0 else f'needs {ri_needed*100:.1f}%'
        print(f'  {node:10s}: payoff@0={p0:+.3f}  {status}')
    print()

print(f'Total minimum budget needed: {total_min*100:.1f}%')
if total_min > 1.0:
    print(f'Exceeds 100%. Nash Equilibrium k* will be > 0.')
    print('Solver finds best possible within budget.')
else:
    print(f'Within 100%. Nash Equilibrium k* can be ≤ 0.')
    print('Defender can win completely.')

# ============================================================
# STAGE 6 — STACKELBERG SOLVER
# ============================================================

print()
print('='*65)
print('STAGE 6 — STACKELBERG SOLVER')
print('Defender finds r* — Nash Equilibrium')
print()
print('WHY STACKELBERG:')
print('  Defender moves FIRST (sets up security)')
print('  Attacker OBSERVES (does reconnaissance)')
print('  Attacker RESPONDS (picks weakest server)')
print()
print('BACKWARD INDUCTION:')
print('  Defender thinks: if I set r, Ravi picks')
print('  max payoff combo. I choose r to minimise')
print('  that maximum.')
print('='*65)
print()
print(f'Active attacks: {active_atks}')
print()

r0_equal = np.ones(N)/N
p_start  = max_payoff_fn(r0_equal, ml_probs, active_atks)
bn0, ba0, bp0 = best_attack_fn(r0_equal, ml_probs, active_atks)

print(f'At equal allocation:')
print(f'  Ravi best payoff : {p_start:.4f}')
print(f'  Ravi best attack : {ba0} on {bn0}')
print()
print('Solving...')

r_star, converged = solve_stackelberg(ml_probs, active_atks)
print(f'Solver converged: {converged}')

# ============================================================
# STAGE 7 — NASH EQUILIBRIUM RESULT
# ============================================================

print()
print('='*65)
print('STAGE 7 — NASH EQUILIBRIUM r*')
print('Optimal security allocation')
print('='*65)
print()

sorted_r = sorted(enumerate(r_star), key=lambda x: x[1], reverse=True)

print('DEPLOY THIS SECURITY ALLOCATION NOW:')
print()
print(f'  {"Rank":>4}  {"Node":10s}  {"r*":>8s}  {"Budget%":>8s}  Bar')
print('  ' + '-'*55)

for rank, (i, alloc) in enumerate(sorted_r, 1):
    node = NODES[i]
    pct  = alloc*100
    bar  = '█' * int(pct)
    print(f'  {rank:>4}  {node:10s}  {alloc:>8.4f}  {pct:>7.2f}%  {bar}')

print()
print(f'  Sum = {r_star.sum()*100:.4f}%')

# ============================================================
# STAGE 8 — VERIFY ALL 20 PAYOFFS AT r*
# ============================================================

print()
print('='*65)
print('STAGE 8 — VERIFY ALL 20 PAYOFFS AT r*')
print('All should be negative')
print('='*65)
print()
print(f'  {"Node":10s}  {"V":>3s}  ' +
      '  '.join(f'{a:>12s}' for a in ATTACK_TYPES))
print('  ' + '-'*75)

all_neg = True
for i, node in enumerate(NODES):
    row = f'  {node:10s}  {NODE_VALUES[node]:>3d}  '
    for atk in ATTACK_TYPES:
        p = payoff(node, atk, r_star[i], ml_probs)
        if auto_neg[atk]:
            row += f'{"auto("+str(round(p,2))+")":>14s}  '
        elif p > 0:
            row    += f'{p:>+14.3f}  '
            all_neg = False
        else:
            row    += f'{p:>14.3f}  '
    print(row)

best_p = max_payoff_fn(r_star, ml_probs, active_atks)
print()
print(f'Best attacker payoff at r* : {best_p:.6f}')

if all_neg and best_p <= 0:
    print()
    print('ALL 20 PAYOFFS NEGATIVE.')
    print('Nash Equilibrium achieved.')
    print('Ravi cannot profit from ANY attack.')
    print('DEFENDER WINS COMPLETELY.')
elif best_p <= 0.5:
    print()
    print('Near Nash Equilibrium.')
    print('Practically optimal within budget.')
else:
    print()
    print(f'Best payoff = {best_p:.4f}')
    print('Budget constraint prevents full neutralisation.')
    print('This is the theoretical minimum achievable.')

# ============================================================
# STAGE 9 — COMPARISON TABLE
# ============================================================

print()
print('='*65)
print('STAGE 9 — STRATEGY COMPARISON')
print('='*65)
print()

r_equal_vec = np.ones(N)/N
v_arr       = np.array([NODE_VALUES[n] for n in NODES], dtype=float)
r_value     = v_arr/v_arr.sum()

print(f'  {"Strategy":30s}  {"Ravi best payoff":>16s}  {"Result"}')
print('  ' + '-'*65)

for name, rv in [
    ('Equal allocation (20% each)', r_equal_vec),
    ('Value-weighted allocation',   r_value),
    ('Stackelberg r* (our model)',  r_star),
]:
    p  = max_payoff_fn(rv, ml_probs, active_atks)
    ok = 'DEFENDER WINS' if p <= 0 else f'Ravi gets {p:.3f}'
    print(f'  {name:30s}  {p:>16.6f}  {ok}')

# ============================================================
# STAGE 10 — GENERATE GRAPHS
# ============================================================

print()
print('Generating graphs...')

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Graph 1 — ML probabilities
ax    = axes[0]
cols  = ['#E53935' if not auto_neg[a] else '#90CAF9' for a in ATTACK_TYPES]
bars  = ax.bar(ATTACK_TYPES,
               [ml_probs[a]*100 for a in ATTACK_TYPES],
               color=cols, alpha=0.87, width=0.5)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h+0.5,
            f'{h:.2f}%', ha='center', va='bottom',
            fontsize=11, fontweight='bold')
ax.set_title(f'ML Output — Record {TEST_INDEX}\n'
             f'True: {true_label} | Predicted: {pred_label}\n'
             f'Red=Active Threat  Blue=Auto Negative',
             fontsize=11)
ax.set_ylabel('Probability (%)')
ax.set_xlabel('Attack Type')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(ml_probs.values())*130 + 5)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#E53935', label='Active threat — needs resource'),
    Patch(color='#90CAF9', label='Auto negative — already safe')
], fontsize=9)

# Graph 2 — r* vs equal allocation
ax = axes[1]
x  = np.arange(N)
w  = 0.35
ax.bar(x-w/2, [1/N*100]*N, w,
       label='Equal allocation', color='#EF9A9A', alpha=0.85)
ax.bar(x+w/2, [ri*100 for ri in r_star], w,
       label='Stackelberg r*', color='#43A047', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(NODES, fontsize=10)
ax.set_title(f'Security Allocation\nEqual vs Nash Equilibrium r*',
             fontsize=12)
ax.set_ylabel('Security Budget %')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Graph 3 — Payoff heatmap at r*
ax = axes[2]
pm = []
for i, node in enumerate(NODES):
    row = []
    for atk in ATTACK_TYPES:
        p = payoff(node, atk, r_star[i], ml_probs)
        row.append(p)
    pm.append(row)
pm  = np.array(pm)
im  = ax.imshow(pm, cmap='RdYlGn_r', aspect='auto', vmin=-3, vmax=1)
plt.colorbar(im, ax=ax, label='Attacker Payoff')
ax.set_xticks(range(4))
ax.set_xticklabels(ATTACK_TYPES)
ax.set_yticks(range(N))
ax.set_yticklabels(NODES)
ax.set_title('Payoff Heatmap at r*\nGreen=Safe  Red=Danger', fontsize=11)
for i in range(N):
    for j, atk in enumerate(ATTACK_TYPES):
        val = pm[i, j]
        txt = f'{val:.2f}'
        clr = 'white' if abs(val) > 1.5 else 'black'
        ax.text(j, i, txt, ha='center', va='center',
                fontsize=8, fontweight='bold', color=clr)

plt.suptitle(f'Complete Pipeline — Record {TEST_INDEX}\n'
             f'True: {true_label} | Predicted: {pred_label} ({confidence*100:.1f}%)',
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'game_result.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: game_result.png')

# ============================================================
# FINAL COMPLETE SUMMARY
# ============================================================

print()
print('='*65)
print('COMPLETE PIPELINE SUMMARY')
print('='*65)
print()
print(f'Record index  : {TEST_INDEX}')
print(f'True label    : {true_label}')
print(f'ML predicted  : {pred_label} ({confidence*100:.2f}%) — {status}')
print()
print('Attack probabilities:')
for atk in ATTACK_TYPES:
    p   = ml_probs[atk]
    tag = '← ACTIVE THREAT' if not auto_neg[atk] else '← auto negative'
    print(f'  {atk:8s}: {p*100:.4f}%  {tag}')
print()
print('Nash Equilibrium r*:')
for rank, (i, alloc) in enumerate(sorted_r, 1):
    pct = alloc*100
    bar = '█' * int(pct)
    print(f'  {rank}. {NODES[i]:10s}: {pct:.2f}%  {bar}')
print()
print(f'Ravi best payoff at r*: {best_p:.6f}')
if best_p <= 0:
    print('DEFENDER WINS — Nash Equilibrium achieved')
print()
print('Graph saved: game_result.png')
print()
print('To test other records change TEST_INDEX:')
print('  0      = DoS attack')
print('  60000  = Normal traffic')
print('  120000 = Probe attack')
print('  135000 = R2L attack')
print('  148400 = U2R attack')
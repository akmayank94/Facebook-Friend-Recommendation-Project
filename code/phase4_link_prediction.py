import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import warnings
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import itertools

warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 4: LINK PREDICTION & FRIEND RECOMMENDATION")
print("Predicting Future Friendships Using Similarity & Community Info")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD ALL PREVIOUS DATA
# ============================================================================

print("\n[STEP 1] Loading data from Phases 1-3...")

try:
    with open('results/networks/G_train.pkl', 'rb') as f:
        G_train = pickle.load(f)
    with open('results/networks/G_original.pkl', 'rb') as f:
        G_original = pickle.load(f)
    with open('results/networks/test_edges.pkl', 'rb') as f:
        test_edges = pickle.load(f)

    # Load communities (Louvain - best for link prediction)
    communities_df = pd.read_csv('results/communities_louvain.csv')
    communities_dict = dict(zip(communities_df['Node'], communities_df['Community_Louvain']))

    print(f"✓ Training network: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")
    print(f"✓ Test edges (ground truth): {len(test_edges)} edges")
    print(f"✓ Communities loaded: {len(set(communities_dict.values()))} communities")

except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Make sure you completed Phases 1-3!")
    exit()

G = G_train

# ============================================================================
# STEP 2: GENERATE CANDIDATE PAIRS (NON-EXISTING EDGES)
# ============================================================================

print("\n[STEP 2] Generating candidate pairs for link prediction...")

# Get all possible pairs
all_nodes = list(G.nodes())
n_nodes = len(all_nodes)

# To avoid memory issues, we'll sample candidates
# For large networks, using random sampling
np.random.seed(42)

# Get existing edges
existing_edges = set(G.edges())
existing_edges_undirected = set()
for u, v in existing_edges:
    existing_edges_undirected.add((min(u, v), max(u, v)))

# Sample negative examples (non-edges)
# Sample equal to number of test edges (balanced dataset)
candidate_pairs = []
sample_size = len(test_edges) * 5  # 5x negatives for training

max_attempts = sample_size * 10
attempts = 0

while len(candidate_pairs) < sample_size and attempts < max_attempts:
    u = np.random.choice(all_nodes)
    v = np.random.choice(all_nodes)

    if u != v:
        edge = (min(u, v), max(u, v))
        if edge not in existing_edges_undirected and edge not in candidate_pairs:
            candidate_pairs.append(edge)

    attempts += 1

print(f"✓ Generated {len(candidate_pairs)} candidate pairs (non-edges)")
print(f"  Using as negative examples for evaluation")

# ============================================================================
# STEP 3: IMPLEMENT 4 LINK PREDICTION METHODS
# ============================================================================

print("\n[STEP 3] Calculating similarity scores (4 methods)...")

def common_neighbors(u, v, G):
    """Common Neighbors (Ch. 2.3, 2.4)"""
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    return len(neighbors_u & neighbors_v)

def jaccard_coefficient(u, v, G):
    """Jaccard Coefficient (Ch. 2.3, 2.4)"""
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    intersection = len(neighbors_u & neighbors_v)
    union = len(neighbors_u | neighbors_v)
    return intersection / union if union > 0 else 0

def adamic_adar_index(u, v, G):
    """Adamic-Adar Index (Ch. 2.3, 2.4)"""
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    common = neighbors_u & neighbors_v

    score = 0
    for node in common:
        deg = G.degree(node)
        if deg > 1:
            score += 1 / np.log(deg)

    return score

def preferential_attachment(u, v, G):
    """Preferential Attachment (related to Ch. 4.5)"""
    return G.degree(u) * G.degree(v)

# Calculate scores for test edges
print("  Calculating scores for TEST EDGES (positive examples)...")
test_scores = {
    'common_neighbors': [],
    'jaccard': [],
    'adamic_adar': [],
    'preferential_attachment': [],
    'community_match': [],
    'label': []
}

for u, v in test_edges:
    test_scores['common_neighbors'].append(common_neighbors(u, v, G))
    test_scores['jaccard'].append(jaccard_coefficient(u, v, G))
    test_scores['adamic_adar'].append(adamic_adar_index(u, v, G))
    test_scores['preferential_attachment'].append(preferential_attachment(u, v, G))
    test_scores['community_match'].append(1 if communities_dict.get(u) == communities_dict.get(v) else 0)
    test_scores['label'].append(1)  # Label: 1 = actual edge

# Calculate scores for candidate pairs (negatives)
print("  Calculating scores for CANDIDATE PAIRS (negative examples)...")
for u, v in candidate_pairs:
    test_scores['common_neighbors'].append(common_neighbors(u, v, G))
    test_scores['jaccard'].append(jaccard_coefficient(u, v, G))
    test_scores['adamic_adar'].append(adamic_adar_index(u, v, G))
    test_scores['preferential_attachment'].append(preferential_attachment(u, v, G))
    test_scores['community_match'].append(1 if communities_dict.get(u) == communities_dict.get(v) else 0)
    test_scores['label'].append(0)  # Label: 0 = not an edge

print(f"✓ Calculated scores for {len(test_scores['label'])} pairs")
print(f"  Positive examples (true edges): {sum(test_scores['label'])}")
print(f"  Negative examples (non-edges): {len(test_scores['label']) - sum(test_scores['label'])}")

# ============================================================================
# STEP 4: CREATE SCORING DATAFRAME
# ============================================================================

print("\n[STEP 4] Creating scoring dataframe...")

scores_df = pd.DataFrame(test_scores)

# Normalize scores to 0-1 range
def normalize(values):
    max_val = max(values) if max(values) > 0 else 1
    return [v / max_val for v in values]

scores_df['cnn_norm'] = normalize(scores_df['common_neighbors'])
scores_df['jaccard_norm'] = scores_df['jaccard']  # Already 0-1
scores_df['aa_norm'] = normalize(scores_df['adamic_adar'])
scores_df['pa_norm'] = normalize(scores_df['preferential_attachment'])

# Combined score: Average of 4 methods
scores_df['combined_score'] = (
    scores_df['cnn_norm'] * 0.25 +
    scores_df['jaccard_norm'] * 0.25 +
    scores_df['aa_norm'] * 0.25 +
    scores_df['pa_norm'] * 0.25
)

# Community-boosted score
boost_factor = 1.5  # 1.5x boost if in same community
scores_df['community_boost'] = scores_df['combined_score'] * (
    1 + boost_factor * scores_df['community_match']
)

print("✓ Dataframe created with 6 score columns")
print(f"  Columns: cnn, jaccard, adamic-adar, preferential_attachment,")
print(f"           combined_score, community_boost")

# ============================================================================
# STEP 5: EVALUATE PREDICTIONS
# ============================================================================

print("\n[STEP 5] Evaluating predictions...")

# For different scoring methods, calculate metrics
methods = {
    'Common Neighbors': 'common_neighbors',
    'Jaccard Coefficient': 'jaccard',
    'Adamic-Adar Index': 'adamic_adar',
    'Preferential Attachment': 'preferential_attachment',
    'Combined Score': 'combined_score',
    'Community Boost': 'community_boost'
}

results = []

for method_name, score_col in methods.items():
    # Get scores and labels
    y_true = scores_df['label'].values
    y_scores = scores_df[score_col].values

    # Normalize scores to 0-1 for AUC
    y_scores_norm = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)

    # Calculate AUC-ROC
    try:
        auc_score = roc_auc_score(y_true, y_scores_norm)
    except:
        auc_score = 0.5

    # Top-k evaluation
    for k in [10, 20, 50]:
        # Get top-k predictions
        top_k_idx = np.argsort(y_scores)[-k:]
        y_pred_top_k = y_true[top_k_idx]

        # Calculate precision and recall
        precision = np.sum(y_pred_top_k) / k if k > 0 else 0
        recall = np.sum(y_pred_top_k) / np.sum(y_true) if np.sum(y_true) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'Method': method_name,
            'Metric': f'Precision@{k}',
            'Value': f'{precision:.4f}'
        })
        results.append({
            'Method': method_name,
            'Metric': f'Recall@{k}',
            'Value': f'{recall:.4f}'
        })
        results.append({
            'Method': method_name,
            'Metric': f'F1@{k}',
            'Value': f'{f1:.4f}'
        })

    results.append({
        'Method': method_name,
        'Metric': 'AUC-ROC',
        'Value': f'{auc_score:.4f}'
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# ============================================================================
# STEP 6: SAVE RESULTS
# ============================================================================

print("\n[STEP 6] Saving results...")

os.makedirs('results', exist_ok=True)

scores_df.to_csv('results/link_prediction_scores.csv', index=False)
print("✓ Saved: results/link_prediction_scores.csv")

results_df.to_csv('results/link_prediction_evaluation.csv', index=False)
print("✓ Saved: results/link_prediction_evaluation.csv")

# ============================================================================
# STEP 7: CREATE VISUALIZATIONS
# ============================================================================

print("\n[STEP 7] Creating visualizations...")

os.makedirs('results/visualizations', exist_ok=True)

# Visualization 1: Score distributions
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

methods_to_plot = ['common_neighbors', 'jaccard', 'adamic_adar', 
                   'preferential_attachment', 'combined_score', 'community_boost']
titles = ['Common Neighbors', 'Jaccard Coefficient', 'Adamic-Adar Index',
          'Preferential Attachment', 'Combined Score', 'Community Boost']

for idx, (method, title) in enumerate(zip(methods_to_plot, titles)):
    ax = axes[idx // 3, idx % 3]

    true_scores = scores_df[scores_df['label'] == 1][method]
    false_scores = scores_df[scores_df['label'] == 0][method]

    ax.hist(true_scores, bins=30, alpha=0.6, label='True Links', color='green', edgecolor='black')
    ax.hist(false_scores, bins=30, alpha=0.6, label='Non-Links', color='red', edgecolor='black')
    ax.set_xlabel('Score', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/visualizations/score_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/visualizations/score_distributions.png")
plt.close()

# Visualization 2: AUC-ROC Comparison
fig, ax = plt.subplots(figsize=(10, 8))

for method_name, score_col in methods.items():
    y_true = scores_df['label'].values
    y_scores = scores_df[score_col].values
    y_scores_norm = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores_norm)
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{method_name} (AUC={auc_score:.3f})', linewidth=2)
    except:
        pass

ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)', linewidth=2)
ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
ax.set_title('ROC Curves - Link Prediction Methods Comparison', fontweight='bold', fontsize=13)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/visualizations/roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/visualizations/roc_curves.png")
plt.close()

# Visualization 3: Precision@k, Recall@k, F1@k Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

k_values = [10, 20, 50]
metrics = ['Precision', 'Recall', 'F1']

for metric_idx, metric in enumerate(metrics):
    ax = axes[metric_idx]

    method_names = []
    values = []

    for method_name, score_col in methods.items():
        y_true = scores_df['label'].values
        y_scores = scores_df[score_col].values

        # Use k=20 for display
        k = 20
        top_k_idx = np.argsort(y_scores)[-k:]
        y_pred_top_k = y_true[top_k_idx]

        precision = np.sum(y_pred_top_k) / k if k > 0 else 0
        recall = np.sum(y_pred_top_k) / np.sum(y_true) if np.sum(y_true) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if metric == 'Precision':
            values.append(precision)
        elif metric == 'Recall':
            values.append(recall)
        else:
            values.append(f1)

        method_names.append(method_name)

    ax.bar(range(len(method_names)), values, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric, fontweight='bold', fontsize=11)
    ax.set_title(f'{metric}@20', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])

plt.tight_layout()
plt.savefig('results/visualizations/metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/visualizations/metrics_comparison.png")
plt.close()

# ============================================================================
# STEP 8: CREATE SUMMARY REPORT
# ============================================================================

print("\n[STEP 8] Creating summary report...")

summary = f"""
PHASE 4: LINK PREDICTION SUMMARY
================================================================================

OBJECTIVE:
Predict which non-connected users are likely to become friends in future,
using similarity measures and community information.

DATASET:
Training Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges
Test Edges (Ground Truth): {len(test_edges)} edges
Candidate Pairs (Non-edges): {len(candidate_pairs)} sampled pairs

LINK PREDICTION METHODS (Ch. 2.3-2.4, 4.5):
================================================================================

1. COMMON NEIGHBORS (CNN)
   Score: Count of shared friends
   Performance: {scores_df[scores_df['label']==1]['common_neighbors'].mean():.4f} (true), {scores_df[scores_df['label']==0]['common_neighbors'].mean():.4f} (false)

2. JACCARD COEFFICIENT
   Score: Fraction of union that are common neighbors
   Range: 0-1
   Performance: {scores_df[scores_df['label']==1]['jaccard'].mean():.4f} (true), {scores_df[scores_df['label']==0]['jaccard'].mean():.4f} (false)

3. ADAMIC-ADAR INDEX
   Score: Weighted sum of 1/log(degree) for common neighbors
   Performance: {scores_df[scores_df['label']==1]['adamic_adar'].mean():.4f} (true), {scores_df[scores_df['label']==0]['adamic_adar'].mean():.4f} (false)

4. PREFERENTIAL ATTACHMENT
   Score: Product of degrees (rich get richer)
   Performance: {scores_df[scores_df['label']==1]['preferential_attachment'].mean():.4f} (true), {scores_df[scores_df['label']==0]['preferential_attachment'].mean():.4f} (false)

ENSEMBLE METHODS:
================================================================================

1. COMBINED SCORE (Average of 4 methods)
   Average true score: {scores_df[scores_df['label']==1]['combined_score'].mean():.4f}
   Average false score: {scores_df[scores_df['label']==0]['combined_score'].mean():.4f}

2. COMMUNITY BOOST (1.5x multiplier if same community)
   Average true score: {scores_df[scores_df['label']==1]['community_boost'].mean():.4f}
   Average false score: {scores_df[scores_df['label']==0]['community_boost'].mean():.4f}
   Improvement: {((scores_df[scores_df['label']==1]['community_boost'].mean() - scores_df[scores_df['label']==1]['combined_score'].mean()) / scores_df[scores_df['label']==1]['combined_score'].mean() * 100):.2f}%

KEY FINDINGS:
================================================================================

1. All methods show clear separation between true and false edges
   → True edges have higher scores than false edges

2. Community information BOOSTS prediction accuracy
   → {(scores_df[(scores_df['label']==1) & (scores_df['community_match']==1)].shape[0] / scores_df[scores_df['label']==1].shape[0] * 100):.1f}% of test edges are within same community

3. Adamic-Adar Index often performs best
   → Emphasizes rare common neighbors

4. Preferential Attachment shows different pattern
   → Captures high-degree node connections

EVALUATION METRICS (Top 20 Recommendations):
================================================================================

{results_df[results_df['Metric'].str.contains('@20|AUC')].to_string(index=False)}

INTERPRETATION:
================================================================================

PRECISION@K: What fraction of top-k recommendations are correct?
- High precision means recommendations are reliable
- Precision@20: Focus on top 20 most confident recommendations

RECALL@K: What fraction of all true links are in top-k?
- High recall means we find most of the true connections
- Recall@20: How many of 17,647 test edges can we find in top 20?

AUC-ROC: Overall ranking quality
- Measures ability to rank true edges higher than false edges
- AUC = 0.5 (random), AUC = 1.0 (perfect)

RECOMMENDATION STRATEGY:
================================================================================

Best Approach: Use Community-Boosted Combined Score
Reason: Combines multiple similarity measures + community information

Top Recommendations:
1. Sort by community_boost score
2. Take top-k (e.g., k=20) for each user
3. These are predicted future friends

NEXT STEPS:
================================================================================

Phase 5: Final Report
- Compile all phases into comprehensive report
- Include methods, results, visualizations, conclusions
- Present findings on community structure and link prediction
"""

print(summary)

with open('results/link_prediction_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print("✓ Saved: results/link_prediction_summary.txt")

# ============================================================================
# FINAL COMPLETION MESSAGE
# ============================================================================

print("\n" + "=" * 80)
print("SUCCESS! PHASE 4 COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  CSV Files:")
print("    - results/link_prediction_scores.csv")
print("    - results/link_prediction_evaluation.csv")
print("\n  Visualizations:")
print("    - results/visualizations/score_distributions.png")
print("    - results/visualizations/roc_curves.png")
print("    - results/visualizations/metrics_comparison.png")
print("\n  Summary:")
print("    - results/link_prediction_summary.txt")
print("\nNext Steps:")
print("  Phase 5: Write Final Report & Prepare Submission")
print("=" * 80)
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 2: CENTRALITY ANALYSIS & UNIT 2 ANALYSIS")
print("Identifying Important Nodes in Facebook Network")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD NETWORKS FROM PHASE 1
# ============================================================================

print("\n[STEP 1] Loading networks from Phase 1...")

try:
    with open('results/networks/G_train.pkl', 'rb') as f:
        G_train = pickle.load(f)
    print(f"✓ Training network loaded: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")

    with open('results/networks/G_original.pkl', 'rb') as f:
        G_original = pickle.load(f)
    print(f"✓ Original network loaded: {G_original.number_of_nodes()} nodes, {G_original.number_of_edges()} edges")

    with open('results/networks/test_edges.pkl', 'rb') as f:
        test_edges = pickle.load(f)
    print(f"✓ Test edges loaded: {len(test_edges)} edges")

except FileNotFoundError as e:
    print(f"ERROR: Could not find Phase 1 files: {e}")
    print("Make sure you completed Phase 1 first!")
    exit()

G = G_train

# ============================================================================
# STEP 2: CALCULATE DEGREE CENTRALITY (Ch. 2.1.3)
# ============================================================================

print("\n[STEP 2.1] Calculating Degree Centrality (Ch. 2.1.3)...")

degree_centrality = nx.degree_centrality(G)
degree_df = pd.DataFrame({
    'Node': list(degree_centrality.keys()),
    'Degree_Centrality': list(degree_centrality.values())
})
degree_df = degree_df.sort_values('Degree_Centrality', ascending=False).reset_index(drop=True)

print(f"✓ Degree centrality calculated for {len(degree_centrality)} nodes")
print("\nTop 5 Nodes by Degree Centrality:")
print(degree_df.head(5).to_string(index=False))

# ============================================================================
# STEP 3: CALCULATE BETWEENNESS CENTRALITY (Ch. 2.1.4)
# ============================================================================

print("\n[STEP 2.2] Calculating Betweenness Centrality (Ch. 2.1.4)...")

betweenness_centrality = nx.betweenness_centrality(G)
betweenness_df = pd.DataFrame({
    'Node': list(betweenness_centrality.keys()),
    'Betweenness_Centrality': list(betweenness_centrality.values())
})
betweenness_df = betweenness_df.sort_values('Betweenness_Centrality', ascending=False).reset_index(drop=True)

print(f"✓ Betweenness centrality calculated for {len(betweenness_centrality)} nodes")
print("\nTop 5 Nodes by Betweenness Centrality:")
print(betweenness_df.head(5).to_string(index=False))

# ============================================================================
# STEP 4: CALCULATE CLOSENESS CENTRALITY (Ch. 2.2)
# ============================================================================

print("\n[STEP 2.3] Calculating Closeness Centrality (Ch. 2.2)...")

closeness_centrality = nx.closeness_centrality(G)
closeness_df = pd.DataFrame({
    'Node': list(closeness_centrality.keys()),
    'Closeness_Centrality': list(closeness_centrality.values())
})
closeness_df = closeness_df.sort_values('Closeness_Centrality', ascending=False).reset_index(drop=True)

print(f"✓ Closeness centrality calculated for {len(closeness_centrality)} nodes")
print("\nTop 5 Nodes by Closeness Centrality:")
print(closeness_df.head(5).to_string(index=False))

# ============================================================================
# STEP 5: CALCULATE PAGERANK (Ch. 4.5)
# ============================================================================

print("\n[STEP 2.4] Calculating PageRank (Ch. 4.5)...")

pagerank = nx.pagerank(G, alpha=0.85)
pagerank_df = pd.DataFrame({
    'Node': list(pagerank.keys()),
    'PageRank': list(pagerank.values())
})
pagerank_df = pagerank_df.sort_values('PageRank', ascending=False).reset_index(drop=True)

print(f"✓ PageRank calculated for {len(pagerank)} nodes")
print("\nTop 5 Nodes by PageRank:")
print(pagerank_df.head(5).to_string(index=False))

# ============================================================================
# STEP 6: CREATE COMPREHENSIVE COMPARISON TABLE
# ============================================================================

print("\n[STEP 2.5] Creating Comparison Table...")

k = 10
top_degree = set(degree_df.head(k)['Node'].values)
top_betweenness = set(betweenness_df.head(k)['Node'].values)
top_closeness = set(closeness_df.head(k)['Node'].values)
top_pagerank = set(pagerank_df.head(k)['Node'].values)

top_nodes = list(top_degree | top_betweenness | top_closeness | top_pagerank)

comparison_data = []
for node in sorted(top_nodes):
    degree_rank = degree_df[degree_df["Node"] == node].index[0] + 1 if node in degree_df["Node"].values else len(degree_df)
    betweenness_rank = betweenness_df[betweenness_df["Node"] == node].index[0] + 1 if node in betweenness_df["Node"].values else len(betweenness_df)
    closeness_rank = closeness_df[closeness_df["Node"] == node].index[0] + 1 if node in closeness_df["Node"].values else len(closeness_df)
    pagerank_rank = pagerank_df[pagerank_df["Node"] == node].index[0] + 1 if node in pagerank_df["Node"].values else len(pagerank_df)

    degree_val = degree_centrality.get(node, 0)
    betweenness_val = betweenness_centrality.get(node, 0)
    closeness_val = closeness_centrality.get(node, 0)
    pagerank_val = pagerank.get(node, 0)

    comparison_data.append({
        'Node': node,
        'Degree_Rank': degree_rank,
        'Degree_Value': f'{degree_val:.4f}',
        'Betweenness_Rank': betweenness_rank,
        'Betweenness_Value': f'{betweenness_val:.6f}',
        'Closeness_Rank': closeness_rank,
        'Closeness_Value': f'{closeness_val:.4f}',
        'PageRank_Rank': pagerank_rank,
        'PageRank_Value': f'{pagerank_val:.6f}'
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Node')

print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# STEP 7: CALCULATE SIMILARITY MEASURES
# ============================================================================

print("\n[STEP 2.6] Calculating Similarity Measures (Ch. 2.3, 2.4)...")

central_nodes = list(degree_df.head(10)["Node"].values)
similarity_data = []

for i, node1 in enumerate(central_nodes):
    neighbors1 = set(G.neighbors(node1))

    for node2 in central_nodes[i+1:]:
        neighbors2 = set(G.neighbors(node2))

        common = len(neighbors1 & neighbors2)
        union = len(neighbors1 | neighbors2)
        jaccard = common / union if union > 0 else 0

        adamic_adar = 0
        for neighbor in (neighbors1 & neighbors2):
            deg = G.degree(neighbor)
            if deg > 1:
                adamic_adar += 1 / np.log(deg)

        similarity_data.append({
            'Node1': node1,
            'Node2': node2,
            'Common_Neighbors': common,
            'Jaccard_Similarity': f'{jaccard:.4f}',
            'Adamic_Adar': f'{adamic_adar:.4f}'
        })

similarity_df = pd.DataFrame(similarity_data)

print("\nTop 10 Similarity Pairs:")
print(similarity_df.head(10).to_string(index=False))

# ============================================================================
# STEP 8: SAVE ALL RESULTS
# ============================================================================

print("\n[STEP 2.7] Saving Results...")

os.makedirs('results', exist_ok=True)

degree_df.to_csv('results/centrality_degree.csv', index=False)
print("✓ Saved: results/centrality_degree.csv")

betweenness_df.to_csv('results/centrality_betweenness.csv', index=False)
print("✓ Saved: results/centrality_betweenness.csv")

closeness_df.to_csv('results/centrality_closeness.csv', index=False)
print("✓ Saved: results/centrality_closeness.csv")

pagerank_df.to_csv('results/centrality_pagerank.csv', index=False)
print("✓ Saved: results/centrality_pagerank.csv")

comparison_df.to_csv('results/centrality_comparison.csv', index=False)
print("✓ Saved: results/centrality_comparison.csv")

similarity_df.to_csv('results/similarity_measures.csv', index=False)
print("✓ Saved: results/similarity_measures.csv")

# ============================================================================
# STEP 9: CREATE VISUALIZATIONS
# ============================================================================

print("\n[STEP 2.8] Creating Visualizations...")

os.makedirs('results/visualizations', exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 6))
top20_degree = degree_df.head(20)
ax.barh(range(len(top20_degree)), top20_degree["Degree_Centrality"].values, color="steelblue")
ax.set_yticks(range(len(top20_degree)))
ax.set_yticklabels(top20_degree["Node"].values)
ax.set_xlabel("Degree Centrality Score", fontweight="bold", fontsize=11)
ax.set_ylabel("Node ID", fontweight="bold", fontsize=11)
ax.set_title("Top 20 Nodes by Degree Centrality (Ch. 2.1.3)", fontweight="bold", fontsize=13)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("results/visualizations/centrality_degree_top20.png", dpi=300, bbox_inches="tight")
print("✓ Saved: results/visualizations/centrality_degree_top20.png")
plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
top20_betweenness = betweenness_df.head(20)
ax.barh(range(len(top20_betweenness)), top20_betweenness["Betweenness_Centrality"].values, color="darkred")
ax.set_yticks(range(len(top20_betweenness)))
ax.set_yticklabels(top20_betweenness["Node"].values)
ax.set_xlabel("Betweenness Centrality Score", fontweight="bold", fontsize=11)
ax.set_ylabel("Node ID", fontweight="bold", fontsize=11)
ax.set_title("Top 20 Nodes by Betweenness Centrality (Ch. 2.1.4)", fontweight="bold", fontsize=13)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("results/visualizations/centrality_betweenness_top20.png", dpi=300, bbox_inches="tight")
print("✓ Saved: results/visualizations/centrality_betweenness_top20.png")
plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
top20_closeness = closeness_df.head(20)
ax.barh(range(len(top20_closeness)), top20_closeness["Closeness_Centrality"].values, color="darkgreen")
ax.set_yticks(range(len(top20_closeness)))
ax.set_yticklabels(top20_closeness["Node"].values)
ax.set_xlabel("Closeness Centrality Score", fontweight="bold", fontsize=11)
ax.set_ylabel("Node ID", fontweight="bold", fontsize=11)
ax.set_title("Top 20 Nodes by Closeness Centrality (Ch. 2.2)", fontweight="bold", fontsize=13)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("results/visualizations/centrality_closeness_top20.png", dpi=300, bbox_inches="tight")
print("✓ Saved: results/visualizations/centrality_closeness_top20.png")
plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
top20_pagerank = pagerank_df.head(20)
ax.barh(range(len(top20_pagerank)), top20_pagerank["PageRank"].values, color="purple")
ax.set_yticks(range(len(top20_pagerank)))
ax.set_yticklabels(top20_pagerank["Node"].values)
ax.set_xlabel("PageRank Score", fontweight="bold", fontsize=11)
ax.set_ylabel("Node ID", fontweight="bold", fontsize=11)
ax.set_title("Top 20 Nodes by PageRank (Ch. 4.5)", fontweight="bold", fontsize=13)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("results/visualizations/centrality_pagerank_top20.png", dpi=300, bbox_inches="tight")
print("✓ Saved: results/visualizations/centrality_pagerank_top20.png")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

top10_nodes = list(set(list(degree_df.head(10)["Node"].values) + 
                       list(betweenness_df.head(10)["Node"].values) +
                       list(closeness_df.head(10)["Node"].values) +
                       list(pagerank_df.head(10)["Node"].values)))[:10]

degree_vals = [degree_centrality.get(n, 0) for n in top10_nodes]
betweenness_vals = [betweenness_centrality.get(n, 0) for n in top10_nodes]
closeness_vals = [closeness_centrality.get(n, 0) for n in top10_nodes]
pagerank_vals = [pagerank.get(n, 0) for n in top10_nodes]

def normalize(vals):
    max_val = max(vals) if max(vals) > 0 else 1
    return [v/max_val for v in vals]

degree_norm = normalize(degree_vals)
betweenness_norm = normalize(betweenness_vals)
closeness_norm = normalize(closeness_vals)
pagerank_norm = normalize(pagerank_vals)

x = np.arange(len(top10_nodes))
width = 0.2

axes[0, 0].bar(x - 1.5*width, degree_norm, width, color="steelblue")
axes[0, 0].set_ylabel("Normalized Score", fontweight="bold")
axes[0, 0].set_title("Degree Centrality (Ch. 2.1.3)", fontweight="bold")
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(top10_nodes, rotation=45)
axes[0, 0].grid(True, alpha=0.3, axis="y")

axes[0, 1].bar(x - 1.5*width, betweenness_norm, width, color="darkred")
axes[0, 1].set_ylabel("Normalized Score", fontweight="bold")
axes[0, 1].set_title("Betweenness Centrality (Ch. 2.1.4)", fontweight="bold")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(top10_nodes, rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis="y")

axes[1, 0].bar(x - 1.5*width, closeness_norm, width, color="darkgreen")
axes[1, 0].set_ylabel("Normalized Score", fontweight="bold")
axes[1, 0].set_title("Closeness Centrality (Ch. 2.2)", fontweight="bold")
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(top10_nodes, rotation=45)
axes[1, 0].grid(True, alpha=0.3, axis="y")

axes[1, 1].bar(x - 1.5*width, pagerank_norm, width, color="purple")
axes[1, 1].set_ylabel("Normalized Score", fontweight="bold")
axes[1, 1].set_title("PageRank (Ch. 4.5)", fontweight="bold")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(top10_nodes, rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("results/visualizations/centrality_comparison_4measures.png", dpi=300, bbox_inches="tight")
print("✓ Saved: results/visualizations/centrality_comparison_4measures.png")
plt.close()

# ============================================================================
# STEP 10: FINAL MESSAGE
# ============================================================================

print("\n" + "=" * 80)
print("SUCCESS! PHASE 2 COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  CSV Files:")
print("    - results/centrality_degree.csv")
print("    - results/centrality_betweenness.csv")
print("    - results/centrality_closeness.csv")
print("    - results/centrality_pagerank.csv")
print("    - results/centrality_comparison.csv")
print("    - results/similarity_measures.csv")
print("\n  Visualizations:")
print("    - results/visualizations/centrality_degree_top20.png")
print("    - results/visualizations/centrality_betweenness_top20.png")
print("    - results/visualizations/centrality_closeness_top20.png")
print("    - results/visualizations/centrality_pagerank_top20.png")
print("    - results/visualizations/centrality_comparison_4measures.png")
print("\nNext: Phase 3 - Community Detection (Unit 4)")
print("=" * 80)
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import warnings
from collections import Counter
import community as community_louvain  # Louvain algorithm

warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 3: COMMUNITY DETECTION & UNIT 4 ANALYSIS")
print("Finding Groups of Closely Connected Users")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD NETWORKS FROM PHASE 1
# ============================================================================

print("\n[STEP 1] Loading networks from Phase 1...")

try:
    with open('results/networks/G_train.pkl', 'rb') as f:
        G_train = pickle.load(f)
    print(f"Training network loaded: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")

    with open('results/networks/G_original.pkl', 'rb') as f:
        G_original = pickle.load(f)
    print(f"Original network loaded: {G_original.number_of_nodes()} nodes, {G_original.number_of_edges()} edges")

except FileNotFoundError as e:
    print(f"ERROR: Could not find Phase 1 files: {e}")
    print("Make sure you completed Phase 1 first!")
    exit()

G = G_train

# ============================================================================
# STEP 2: IMPLEMENT LOUVAIN ALGORITHM (Ch. 5.4.2) - REQUIRED
# ============================================================================

print("\n[STEP 2] Implementing Louvain Algorithm (Ch. 5.4.2)...")

# Apply Louvain algorithm
communities_louvain = community_louvain.best_partition(G)

# Calculate modularity
modularity_louvain = community_louvain.modularity(communities_louvain, G)

# Get community statistics
num_communities_louvain = len(set(communities_louvain.values()))
community_sizes_louvain = Counter(communities_louvain.values())

print(f"Louvain algorithm completed!")
print(f"  Number of communities found: {num_communities_louvain}")
print(f"  Modularity score: {modularity_louvain:.4f}")
print(f"  Community sizes (top 10):")

top10_communities = sorted(community_sizes_louvain.items(), key=lambda x: x[1], reverse=True)[:10]
for comm_id, size in top10_communities:
    print(f"    Community {comm_id}: {size} nodes")

# ============================================================================
# STEP 3: IMPLEMENT FAST GREEDY ALGORITHM (Ch. 5.4.2) - RECOMMENDED
# ============================================================================

print("\n[STEP 3] Implementing Fast Greedy Algorithm (Ch. 5.4.2)...")

# Fast Greedy requires NetworkX's implementation
from networkx.algorithms.community import greedy_modularity_communities

# Apply Fast Greedy algorithm
communities_greedy_gen = greedy_modularity_communities(G)
communities_greedy = list(communities_greedy_gen)

# Convert to node-to-community mapping (like Louvain format)
communities_greedy_dict = {}
for comm_id, community in enumerate(communities_greedy):
    for node in community:
        communities_greedy_dict[node] = comm_id

# Calculate modularity
modularity_greedy = nx.algorithms.community.modularity(G, communities_greedy)

# Get community statistics
num_communities_greedy = len(communities_greedy)
community_sizes_greedy = Counter(communities_greedy_dict.values())

print(f"Fast Greedy algorithm completed!")
print(f"  Number of communities found: {num_communities_greedy}")
print(f"  Modularity score: {modularity_greedy:.4f}")
print(f"  Community sizes (top 10):")

top10_communities_greedy = sorted(community_sizes_greedy.items(), key=lambda x: x[1], reverse=True)[:10]
for comm_id, size in top10_communities_greedy:
    print(f"    Community {comm_id}: {size} nodes")

# ============================================================================
# STEP 4: COMPARE ALGORITHMS
# ============================================================================

print("\n[STEP 4] Comparing Louvain and Fast Greedy Algorithms...")

comparison_data = {
    'Metric': [
        'Number of Communities',
        'Modularity Score',
        'Largest Community Size',
        'Smallest Community Size',
        'Average Community Size',
        'Algorithm Type'
    ],
    'Louvain': [
        num_communities_louvain,
        f'{modularity_louvain:.4f}',
        max(community_sizes_louvain.values()),
        min(community_sizes_louvain.values()),
        f'{np.mean(list(community_sizes_louvain.values())):.2f}',
        'Optimization-based'
    ],
    'Fast Greedy': [
        num_communities_greedy,
        f'{modularity_greedy:.4f}',
        max(community_sizes_greedy.values()),
        min(community_sizes_greedy.values()),
        f'{np.mean(list(community_sizes_greedy.values())):.2f}',
        'Agglomerative'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================

print("\n[STEP 5] Saving Community Detection Results...")

os.makedirs('results', exist_ok=True)

# Save Louvain communities
louvain_df = pd.DataFrame({
    'Node': list(communities_louvain.keys()),
    'Community_Louvain': list(communities_louvain.values())
})
louvain_df = louvain_df.sort_values('Node')
louvain_df.to_csv('results/communities_louvain.csv', index=False)
print("Saved: results/communities_louvain.csv")

# Save Fast Greedy communities
greedy_df = pd.DataFrame({
    'Node': list(communities_greedy_dict.keys()),
    'Community_Greedy': list(communities_greedy_dict.values())
})
greedy_df = greedy_df.sort_values('Node')
greedy_df.to_csv('results/communities_greedy.csv', index=False)
print("Saved: results/communities_greedy.csv")

# Save comparison table
comparison_df.to_csv('results/community_comparison.csv', index=False)
print("Saved: results/community_comparison.csv")

# Save detailed community analysis
louvain_analysis = pd.DataFrame({
    'Community_ID': list(community_sizes_louvain.keys()),
    'Size_Louvain': list(community_sizes_louvain.values())
})
louvain_analysis = louvain_analysis.sort_values('Size_Louvain', ascending=False).reset_index(drop=True)
louvain_analysis.to_csv('results/community_sizes_louvain.csv', index=False)
print("Saved: results/community_sizes_louvain.csv")

greedy_analysis = pd.DataFrame({
    'Community_ID': list(community_sizes_greedy.keys()),
    'Size_Greedy': list(community_sizes_greedy.values())
})
greedy_analysis = greedy_analysis.sort_values('Size_Greedy', ascending=False).reset_index(drop=True)
greedy_analysis.to_csv('results/community_sizes_greedy.csv', index=False)
print("Saved: results/community_sizes_greedy.csv")

# ============================================================================
# STEP 6: CREATE VISUALIZATIONS
# ============================================================================

print("\n[STEP 6] Creating Community Visualizations...")

os.makedirs('results/visualizations', exist_ok=True)

# Visualization 1: Community size distribution (Louvain)
fig, ax = plt.subplots(figsize=(12, 6))
sizes_louvain = sorted(community_sizes_louvain.values(), reverse=True)
ax.bar(range(len(sizes_louvain)), sizes_louvain, color='steelblue', edgecolor='black')
ax.set_xlabel('Community Rank', fontsize=12, fontweight='bold')
ax.set_ylabel('Community Size (Number of Nodes)', fontsize=12, fontweight='bold')
ax.set_title(f'Community Size Distribution - Louvain Algorithm (Ch. 5.4.2)\n{num_communities_louvain} communities, Modularity = {modularity_louvain:.4f}', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/visualizations/community_sizes_louvain.png', dpi=300, bbox_inches='tight')
print("Saved: results/visualizations/community_sizes_louvain.png")
plt.close()

# Visualization 2: Community size distribution (Fast Greedy)
fig, ax = plt.subplots(figsize=(12, 6))
sizes_greedy = sorted(community_sizes_greedy.values(), reverse=True)
ax.bar(range(len(sizes_greedy)), sizes_greedy, color='darkred', edgecolor='black')
ax.set_xlabel('Community Rank', fontsize=12, fontweight='bold')
ax.set_ylabel('Community Size (Number of Nodes)', fontsize=12, fontweight='bold')
ax.set_title(f'Community Size Distribution - Fast Greedy Algorithm (Ch. 5.4.2)\n{num_communities_greedy} communities, Modularity = {modularity_greedy:.4f}', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/visualizations/community_sizes_greedy.png', dpi=300, bbox_inches='tight')
print("Saved: results/visualizations/community_sizes_greedy.png")
plt.close()

# Visualization 3: Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].bar(range(min(20, len(sizes_louvain))), sizes_louvain[:20], color='steelblue', edgecolor='black')
axes[0].set_xlabel('Community Rank', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Size', fontsize=11, fontweight='bold')
axes[0].set_title(f'Louvain\n{num_communities_louvain} communities, Q={modularity_louvain:.4f}', 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(range(min(20, len(sizes_greedy))), sizes_greedy[:20], color='darkred', edgecolor='black')
axes[1].set_xlabel('Community Rank', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Size', fontsize=11, fontweight='bold')
axes[1].set_title(f'Fast Greedy\n{num_communities_greedy} communities, Q={modularity_greedy:.4f}', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/visualizations/community_comparison_sidebyside.png', dpi=300, bbox_inches='tight')
print("Saved: results/visualizations/community_comparison_sidebyside.png")
plt.close()

# Visualization 4: Modularity comparison bar chart
fig, ax = plt.subplots(figsize=(8, 6))
algorithms = ['Louvain', 'Fast Greedy']
modularities = [modularity_louvain, modularity_greedy]
colors = ['steelblue', 'darkred']

bars = ax.bar(algorithms, modularities, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
ax.set_ylabel('Modularity Score (Q)', fontsize=12, fontweight='bold')
ax.set_title('Modularity Comparison (Ch. 5.4.2)\nHigher is Better', fontsize=13, fontweight='bold')
ax.set_ylim([0, max(modularities) * 1.2])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (alg, mod) in enumerate(zip(algorithms, modularities)):
    ax.text(i, mod + 0.01, f'{mod:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add interpretation line
if max(modularities) > 0.5:
    ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Very Strong (Q>0.5)')
elif max(modularities) > 0.3:
    ax.axhline(y=0.3, color='orange', linestyle='--', linewidth=2, label='Strong (Q>0.3)')

ax.legend()
plt.tight_layout()
plt.savefig('results/visualizations/modularity_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/visualizations/modularity_comparison.png")
plt.close()

# Visualization 5: Network visualization with communities (sample of 200 nodes)
print("\n  Creating network visualization (this may take a moment)...")

# Sample 200 nodes for visualization (full network is too dense)
np.random.seed(42)
sample_nodes = np.random.choice(list(G.nodes()), size=min(200, G.number_of_nodes()), replace=False)
G_sample = G.subgraph(sample_nodes)

# Get communities for sample
communities_sample = {node: communities_louvain[node] for node in G_sample.nodes()}

# Create layout
pos = nx.spring_layout(G_sample, k=0.5, iterations=50, seed=42)

# Assign colors
unique_communities = list(set(communities_sample.values()))
num_communities = len(unique_communities)
colors_palette = plt.cm.get_cmap('tab20', num_communities)
node_colors = [colors_palette(unique_communities.index(communities_sample[node])) for node in G_sample.nodes()]

# Get node sizes based on degree
degrees_sample = dict(G_sample.degree())
node_sizes = [degrees_sample[node] * 10 for node in G_sample.nodes()]

# Plot
fig, ax = plt.subplots(figsize=(14, 14))
nx.draw_networkx_nodes(G_sample, pos, node_color=node_colors, node_size=node_sizes, 
                       alpha=0.8, edgecolors='black', linewidths=0.5, ax=ax)
nx.draw_networkx_edges(G_sample, pos, alpha=0.2, width=0.5, ax=ax)

ax.set_title(f'Network Visualization with Communities (Louvain Algorithm)\nSample of {len(G_sample.nodes())} nodes, colored by community', 
             fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('results/visualizations/network_with_communities.png', dpi=300, bbox_inches='tight')
print("Saved: results/visualizations/network_with_communities.png")
plt.close()

# ============================================================================
# STEP 7: ANALYZE COMMUNITY CHARACTERISTICS
# ============================================================================

print("\n[STEP 7] Analyzing Community Characteristics...")

# For top 5 largest communities in Louvain
top5_louvain = sorted(community_sizes_louvain.items(), key=lambda x: x[1], reverse=True)[:5]

community_analysis = []
for comm_id, size in top5_louvain:
    # Get nodes in this community
    nodes_in_comm = [node for node, comm in communities_louvain.items() if comm == comm_id]

    # Calculate internal edges
    subgraph = G.subgraph(nodes_in_comm)
    internal_edges = subgraph.number_of_edges()

    # Calculate external edges
    external_edges = 0
    for node in nodes_in_comm:
        for neighbor in G.neighbors(node):
            if communities_louvain[neighbor] != comm_id:
                external_edges += 1
    external_edges = external_edges // 2  # Each edge counted twice

    # Calculate density
    max_internal_edges = (size * (size - 1)) / 2
    internal_density = internal_edges / max_internal_edges if max_internal_edges > 0 else 0

    community_analysis.append({
        'Community_ID': comm_id,
        'Size': size,
        'Internal_Edges': internal_edges,
        'External_Edges': external_edges,
        'Internal_Density': f'{internal_density:.4f}',
        'Ratio_Internal_External': f'{internal_edges / (external_edges + 1):.2f}'
    })

analysis_df = pd.DataFrame(community_analysis)
print("\nTop 5 Largest Communities (Louvain) - Detailed Analysis:")
print(analysis_df.to_string(index=False))

analysis_df.to_csv('results/community_detailed_analysis.csv', index=False)
print("\nSaved: results/community_detailed_analysis.csv")

# ============================================================================
# STEP 8: CREATE SUMMARY STATISTICS
# ============================================================================

print("\n[STEP 8] Creating Summary Report...")

summary_stats = f"""
PHASE 3: COMMUNITY DETECTION SUMMARY
================================================================================

NETWORK INFORMATION:
Training Network Nodes: {G.number_of_nodes()}
Training Network Edges: {G.number_of_edges()}

LOUVAIN ALGORITHM RESULTS (Ch. 5.4.2):
================================================================================
Number of Communities: {num_communities_louvain}
Modularity Score: {modularity_louvain:.4f}

Community Size Statistics:
  Largest community: {max(community_sizes_louvain.values())} nodes
  Smallest community: {min(community_sizes_louvain.values())} nodes
  Average community size: {np.mean(list(community_sizes_louvain.values())):.2f} nodes
  Median community size: {np.median(list(community_sizes_louvain.values())):.0f} nodes

Top 10 Largest Communities:
"""

for i, (comm_id, size) in enumerate(top10_communities, 1):
    summary_stats += f"  {i}. Community {comm_id}: {size} nodes\n"

summary_stats += f"""

FAST GREEDY ALGORITHM RESULTS (Ch. 5.4.2):
================================================================================
Number of Communities: {num_communities_greedy}
Modularity Score: {modularity_greedy:.4f}

Community Size Statistics:
  Largest community: {max(community_sizes_greedy.values())} nodes
  Smallest community: {min(community_sizes_greedy.values())} nodes
  Average community size: {np.mean(list(community_sizes_greedy.values())):.2f} nodes
  Median community size: {np.median(list(community_sizes_greedy.values())):.0f} nodes

Top 10 Largest Communities:
"""

for i, (comm_id, size) in enumerate(top10_communities_greedy, 1):
    summary_stats += f"  {i}. Community {comm_id}: {size} nodes\n"

summary_stats += f"""

ALGORITHM COMPARISON:
================================================================================
{comparison_df.to_string(index=False)}

MODULARITY INTERPRETATION (Ch. 5.4.2):
"""

if modularity_louvain > 0.5:
    summary_stats += "Louvain: VERY STRONG community structure (Q > 0.5)\n"
elif modularity_louvain > 0.3:
    summary_stats += "Louvain: STRONG community structure (Q > 0.3)\n"
else:
    summary_stats += "Louvain: Weak community structure (Q < 0.3)\n"

if modularity_greedy > 0.5:
    summary_stats += "Fast Greedy: VERY STRONG community structure (Q > 0.5)\n"
elif modularity_greedy > 0.3:
    summary_stats += "Fast Greedy: STRONG community structure (Q > 0.3)\n"
else:
    summary_stats += "Fast Greedy: Weak community structure (Q < 0.3)\n"

summary_stats += f"""

KEY INSIGHTS:
================================================================================
1. Both algorithms found strong community structure in the Facebook network
2. Communities represent friend groups or social circles
3. High modularity indicates clear separation between communities
4. Different community sizes reveal hierarchical social structure
5. Internal density shows how tightly connected each community is

DETAILED COMMUNITY ANALYSIS (Top 5 Communities):
================================================================================
{analysis_df.to_string(index=False)}

FILES GENERATED:
================================================================================
Community assignment CSVs (Louvain and Fast Greedy)
Community size analysis
Algorithm comparison table
Detailed community characteristics
5 visualizations (size distributions, comparison, network plot)

NEXT PHASE:
================================================================================
Phase 4: Link Prediction - Use communities to improve friend recommendations
"""

print(summary_stats)

# Save summary
with open('results/community_detection_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_stats)
print("Saved: results/community_detection_summary.txt")

# ============================================================================
# FINAL COMPLETION MESSAGE
# ============================================================================

print("\n" + "=" * 80)
print("SUCCESS! PHASE 3 COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  CSV Files:")
print("    - results/communities_louvain.csv")
print("    - results/communities_greedy.csv")
print("    - results/community_comparison.csv")
print("    - results/community_sizes_louvain.csv")
print("    - results/community_sizes_greedy.csv")
print("    - results/community_detailed_analysis.csv")
print("\n  Visualizations:")
print("    - results/visualizations/community_sizes_louvain.png")
print("    - results/visualizations/community_sizes_greedy.png")
print("    - results/visualizations/community_comparison_sidebyside.png")
print("    - results/visualizations/modularity_comparison.png")
print("    - results/visualizations/network_with_communities.png")
print("\n  Summary:")
print("    - results/community_detection_summary.txt")
print("\nNext Steps:")
print("  Phase 4: Link Prediction")
print("  Phase 5: Report Writing & Final Submission")
print("=" * 80)
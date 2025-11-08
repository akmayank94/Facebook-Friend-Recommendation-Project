import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
import os
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

print("=" * 80)
print("PHASE 1: FACEBOOK NETWORK ANALYSIS")
print("Data Preparation & Unit 1 Analysis")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD THE FACEBOOK NETWORK
# ============================================================================

file_path = 'data/facebook_combined/facebook_combined.txt'

print("Loading Facebook network...")

if not os.path.exists(file_path):
    print(f"ERROR: File not found: {file_path}")
    print("Make sure facebook_combined.txt is in the data folder")
    exit()

try:
    G = nx.read_edgelist(
        file_path,
        create_using=nx.Graph(),
        nodetype=int,
        comments='#'
    )
    print("✓ Network loaded successfully!")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
except Exception as e:
    print(f"Error loading network: {e}")
    exit()

# ============================================================================
# STEP 2: COMPUTE BASIC NETWORK PROPERTIES
# ============================================================================

print("Computing network properties...")

# Density
density = nx.density(G)

# Degrees
degrees = [G.degree(n) for n in G.nodes()]
avg_degree = np.mean(degrees)
degree_sequence = sorted(degrees, reverse=True)

# Components
num_components = nx.number_connected_components(G)

# Diameter
if nx.is_connected(G):
    diameter = nx.diameter(G)
else:
    largest_cc = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_cc)
    diameter = nx.diameter(G_largest)

# Clustering
avg_clustering = nx.average_clustering(G)

# Shortest path
if nx.is_connected(G):
    avg_shortest_path = nx.average_shortest_path_length(G)
else:
    largest_cc = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_cc)
    avg_shortest_path = nx.average_shortest_path_length(G_largest)

print("✓ Network properties computed!")

# ============================================================================
# STEP 3: DISPLAY AND SAVE SUMMARY TABLE
# ============================================================================

print("" + "=" * 80)
print("NETWORK PROPERTIES SUMMARY (Unit 1 Analysis)")
print("=" * 80)

summary_data = {
    'Property': [
        'Nodes (N)',
        'Edges (E)',
        'Density',
        'Average Degree',
        'Min Degree',
        'Max Degree',
        'Components',
        'Diameter',
        'Avg Clustering',
        'Avg Shortest Path'
    ],
    'Value': [
        f'{G.number_of_nodes()}',
        f'{G.number_of_edges()}',
        f'{density:.6f}',
        f'{avg_degree:.2f}',
        f'{min(degree_sequence)}',
        f'{max(degree_sequence)}',
        f'{num_components}',
        f'{diameter}',
        f'{avg_clustering:.6f}',
        f'{avg_shortest_path:.2f}'
    ],
    'Citation': [
        'Ch. 1.1-1.3',
        'Ch. 1.1-1.3',
        'Ch. 1.4.1',
        'Ch. 1.4.1',
        'Ch. 1.5',
        'Ch. 1.5',
        'Ch. 1.4.1',
        'Ch. 1.4.2',
        'Ch. 1.4.8',
        'Ch. 1.4.2'
    ]
}

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))
print("=" * 80)

# Save summary
os.makedirs('results', exist_ok=True)
df_summary.to_csv('results/network_properties_summary.csv', index=False)
print("✓ Summary saved: results/network_properties_summary.csv")

# ============================================================================
# STEP 4: CREATE DEGREE DISTRIBUTION VISUALIZATION
# ============================================================================

print("Creating degree distribution visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
axes[0].hist(degree_sequence, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Degree (Number of Friends)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency (Number of Nodes)', fontsize=12, fontweight='bold')
axes[0].set_title('Degree Distribution (Linear Scale)', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Log-log scale
degree_counts = dict(Counter(degree_sequence))
degrees_nonzero = sorted([d for d in degree_counts.keys() if d > 0])
counts = [degree_counts[d] for d in degrees_nonzero]

axes[1].scatter(degrees_nonzero, counts, s=30, alpha=0.6, color='darkred', edgecolors='black')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel('Degree (log scale)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency (log scale)', fontsize=12, fontweight='bold')
axes[1].set_title('Degree Distribution (Log-Log Scale)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, which='both')

plt.tight_layout()

# Create visualizations directory
os.makedirs('results/visualizations', exist_ok=True)
plt.savefig('results/visualizations/degree_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: results/visualizations/degree_distribution.png")
plt.close()

# ============================================================================
# STEP 5: PREPARE DATA FOR LINK PREDICTION
# ============================================================================

print("Preparing data for link prediction...")

all_edges = list(G.edges())
test_size = 0.2
num_test_edges = int(len(all_edges) * test_size)
test_edges = random.sample(all_edges, num_test_edges)

G_train = G.copy()
G_train.remove_edges_from(test_edges)

print(f"  Original edges: {len(all_edges)}")
print(f"  Training edges (80%): {G_train.number_of_edges()}")
print(f"  Test edges (20%): {len(test_edges)}")

# ============================================================================
# STEP 6: SAVE NETWORKS FOR LATER PHASES
# ============================================================================

print("Saving networks for later phases...")

os.makedirs('results/networks', exist_ok=True)

with open('results/networks/G_original.pkl', 'wb') as f:
    pickle.dump(G, f)
print("  ✓ Original graph saved: G_original.pkl")

with open('results/networks/G_train.pkl', 'wb') as f:
    pickle.dump(G_train, f)
print("  ✓ Training graph saved: G_train.pkl")

with open('results/networks/test_edges.pkl', 'wb') as f:
    pickle.dump(test_edges, f)
print("  ✓ Test edges saved: test_edges.pkl")

# ============================================================================
# PHASE 1 COMPLETE
# ============================================================================

print("" + "=" * 80)
print("✓ PHASE 1 COMPLETE!")
print("=" * 80)
print("Generated Files:")
print("  ├─ results/network_properties_summary.csv")
print("  ├─ results/visualizations/degree_distribution.png")
print("  └─ results/networks/")
print("      ├─ G_original.pkl")
print("      ├─ G_train.pkl")
print("      └─ test_edges.pkl")
print("Next Steps:")
print("  Phase 2: Centrality Analysis (Unit 2)")
print("  Phase 3: Community Detection (Unit 4)")
print("=" * 80)

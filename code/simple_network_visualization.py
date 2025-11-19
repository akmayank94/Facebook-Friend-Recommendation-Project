"""
STANDALONE NETWORK VISUALIZATION - COMMUNITIES WITH ALL NODES
Simple script that creates beautiful community network diagrams
Just run it after Phase 3 is complete!
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from matplotlib.patches import Patch

print("=" * 80)
print("COMMUNITY NETWORK VISUALIZATION - ALL NODES WITH COLORS")
print("=" * 80)

# Load the data
print("\n[1] Loading network and communities...")
with open('results/networks/G_train.pkl', 'rb') as f:
    G = pickle.load(f)

communities_df = pd.read_csv('results/communities_louvain.csv')
communities = dict(zip(communities_df['Node'], communities_df['Community_Louvain']))

print(f"✓ Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"✓ Communities: {len(set(communities.values()))} detected")

# Get unique communities and assign colors
unique_comms = sorted(set(communities.values()))
num_comms = len(unique_comms)

# Create color palette
if num_comms <= 20:
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
else:
    colors = plt.cm.hsv(np.linspace(0, 1, num_comms))

print(f"✓ Color palette created for {num_comms} communities")

# ============================================================================
# VISUALIZATION 1: FULL NETWORK - ALL NODES
# ============================================================================

print("\n[2] Creating FULL NETWORK visualization (all 4,039 nodes)...")
print("    Computing layout... (this takes 2-3 minutes, please wait)")

# Spring layout - spreads nodes based on connections
pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42, scale=10)
print("    ✓ Layout computed!")

# Assign colors to nodes based on community
node_colors = []
for node in G.nodes():
    comm_id = communities[node]
    color_idx = unique_comms.index(comm_id)
    node_colors.append(colors[color_idx])

# Node sizes based on degree
node_degrees = dict(G.degree())
node_sizes = [node_degrees[node] * 2 for node in G.nodes()]

# Create the figure
print("    Drawing network...")
fig, ax = plt.subplots(figsize=(24, 24), facecolor='white')

# Draw edges (light gray, transparent)
nx.draw_networkx_edges(G, pos, alpha=0.08, width=0.3, edge_color='gray', ax=ax)

# Draw nodes (colored by community)
nx.draw_networkx_nodes(G, pos, 
                       node_color=node_colors,
                       node_size=node_sizes,
                       alpha=0.85,
                       edgecolors='black',
                       linewidths=0.2,
                       ax=ax)

# Get top 15 communities by size
comm_sizes = pd.Series(communities.values()).value_counts()
top_comms = comm_sizes.head(15).index.tolist()

# Create legend
legend_elements = []
for comm in top_comms:
    color_idx = unique_comms.index(comm)
    legend_elements.append(
        Patch(facecolor=colors[color_idx], 
              edgecolor='black',
              label=f'Community {comm} ({comm_sizes[comm]} nodes)')
    )

ax.legend(handles=legend_elements, 
         loc='upper left', 
         fontsize=11,
         framealpha=0.95,
         title='Communities (Top 15)',
         title_fontsize=12)

# Title
ax.set_title(
    f'Facebook Network - Community Structure\n'
    f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(unique_comms)} communities\n'
    f'Node colors = communities | Node size = degree (connections)',
    fontsize=18, 
    fontweight='bold', 
    pad=20
)

ax.axis('off')
plt.tight_layout()

print("    Saving image... (this may take a minute)")
plt.savefig('results/visualizations/FULL_NETWORK_COMMUNITIES.png', 
           dpi=300, 
           bbox_inches='tight', 
           facecolor='white',
           edgecolor='none')
print("✓ SAVED: results/visualizations/FULL_NETWORK_COMMUNITIES.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: FORCE-DIRECTED LAYOUT (prettier version)
# ============================================================================

print("\n[3] Creating FORCE-DIRECTED layout (prettier version)...")
print("    Computing layout... (2 minutes)")

# Different algorithm - may look better
pos2 = nx.spring_layout(G, k=0.3, iterations=30, seed=42, scale=20)
print("    ✓ Layout computed!")

fig, ax = plt.subplots(figsize=(24, 24), facecolor='white')

# Draw
nx.draw_networkx_edges(G, pos2, alpha=0.05, width=0.2, edge_color='lightgray', ax=ax)
nx.draw_networkx_nodes(G, pos2, 
                       node_color=node_colors,
                       node_size=node_sizes,
                       alpha=0.9,
                       edgecolors='darkgray',
                       linewidths=0.15,
                       ax=ax)

ax.set_title(
    f'Facebook Network - Communities (Force-Directed Layout)\n'
    f'{G.number_of_nodes()} nodes | {len(unique_comms)} communities',
    fontsize=18, 
    fontweight='bold', 
    pad=20
)

ax.axis('off')
plt.tight_layout()

print("    Saving image...")
plt.savefig('results/visualizations/FORCE_DIRECTED_COMMUNITIES.png',
           dpi=300,
           bbox_inches='tight',
           facecolor='white')
print("✓ SAVED: results/visualizations/FORCE_DIRECTED_COMMUNITIES.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: LARGEST 3 COMMUNITIES IN DETAIL
# ============================================================================

print("\n[4] Creating detailed view of 3 largest communities...")

top_3_comms = comm_sizes.head(3).index.tolist()

fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')

for idx, comm in enumerate(top_3_comms):
    ax = axes[idx]

    # Get nodes in this community
    nodes = [n for n, c in communities.items() if c == comm]

    # Create subgraph
    G_sub = G.subgraph(nodes)

    # Layout for subgraph
    pos_sub = nx.spring_layout(G_sub, k=1, iterations=50, seed=42)

    # Get color for this community
    color_idx = unique_comms.index(comm)

    # Draw
    nx.draw_networkx_edges(G_sub, pos_sub, alpha=0.3, width=0.5, ax=ax)
    nx.draw_networkx_nodes(G_sub, pos_sub,
                          node_color=[colors[color_idx]] * len(G_sub),
                          node_size=[node_degrees[n] * 20 for n in G_sub.nodes()],
                          alpha=1,
                          edgecolors='black',
                          linewidths=1,
                          ax=ax)

    # Statistics
    density = nx.density(G_sub)
    avg_deg = np.mean([d for n, d in G_sub.degree()])

    ax.set_title(
        f'Community {comm}\n'
        f'{len(nodes)} nodes | Density: {density:.3f} | Avg degree: {avg_deg:.1f}',
        fontweight='bold',
        fontsize=12
    )
    ax.axis('off')

plt.suptitle('Top 3 Largest Communities - Internal Structure',
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

print("    Saving image...")
plt.savefig('results/visualizations/TOP3_COMMUNITIES_DETAIL.png',
           dpi=300,
           bbox_inches='tight',
           facecolor='white')
print("✓ SAVED: results/visualizations/TOP3_COMMUNITIES_DETAIL.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUCCESS! NETWORK VISUALIZATIONS COMPLETE!")
print("=" * 80)
print("\nGenerated 3 visualizations:")
print("  ✓ FULL_NETWORK_COMMUNITIES.png (24x24, all 4,039 nodes)")
print("  ✓ FORCE_DIRECTED_COMMUNITIES.png (24x24, different layout)")
print("  ✓ TOP3_COMMUNITIES_DETAIL.png (24x8, zoom on top 3)")
print("\nLocation: results/visualizations/")
print("\nWhat you'll see:")
print("  • Colorful clusters = different communities")
print("  • Different colors clearly show community structure")
print("  • Node size = number of connections")
print("  • Edges show how nodes are connected")
print("\n" + "=" * 80)
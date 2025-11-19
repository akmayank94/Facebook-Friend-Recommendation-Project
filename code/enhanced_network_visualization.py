"""
ENHANCED NETWORK VISUALIZATION WITH COMMUNITIES
Add this to your Phase 3 script OR run as standalone script
Generates beautiful network diagram with communities colored differently
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from matplotlib.patches import Patch

print("="*80)
print("ENHANCED NETWORK VISUALIZATION - COMMUNITY STRUCTURE")
print("="*80)

# Load data
print("\n[STEP 1] Loading network and community data...")

with open('results/networks/G_train.pkl', 'rb') as f:
    G = pickle.load(f)

communities_df = pd.read_csv('results/communities_louvain.csv')
communities_dict = dict(zip(communities_df['Node'], communities_df['Community_Louvain']))

print(f"✓ Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"✓ Communities: {len(set(communities_dict.values()))} detected")

# ============================================================================
# VISUALIZATION 1: FULL NETWORK (ALL 4,039 NODES)
# ============================================================================

print("\n[STEP 2] Creating full network visualization...")
print("  This may take 2-3 minutes for 4,039 nodes...")

# Create layout (use spring layout for organic look)
print("  Computing layout (this is the slow part)...")
pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)
print("  ✓ Layout computed!")

# Get unique communities and create color map
unique_communities = sorted(set(communities_dict.values()))
num_communities = len(unique_communities)

# Use a colorful palette (tab20 for up to 20 colors, then cycle)
if num_communities <= 20:
    colors_palette = plt.cm.tab20(np.linspace(0, 1, 20))
else:
    colors_palette = plt.cm.hsv(np.linspace(0, 1, num_communities))

# Map each node to its community color
node_colors = []
for node in G.nodes():
    comm = communities_dict.get(node, 0)
    color_idx = unique_communities.index(comm) % len(colors_palette)
    node_colors.append(colors_palette[color_idx])

# Get node sizes based on degree (larger = more connections)
degrees = dict(G.degree())
node_sizes = [degrees[node] * 2 for node in G.nodes()]  # Scale factor

# Create figure
fig, ax = plt.subplots(figsize=(20, 20), facecolor='white')

# Draw edges first (in background)
print("  Drawing edges...")
nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5, edge_color='gray', ax=ax)

# Draw nodes on top
print("  Drawing nodes...")
nx.draw_networkx_nodes(G, pos, 
                       node_color=node_colors,
                       node_size=node_sizes,
                       alpha=0.8,
                       edgecolors='black',
                       linewidths=0.3,
                       ax=ax)

# Create legend showing top 10 communities by size
community_sizes = pd.Series(communities_dict.values()).value_counts()
top_10_communities = community_sizes.head(10).index.tolist()

legend_elements = []
for i, comm in enumerate(top_10_communities):
    color_idx = unique_communities.index(comm) % len(colors_palette)
    legend_elements.append(
        Patch(facecolor=colors_palette[color_idx], 
              edgecolor='black',
              label=f'Community {comm} ({community_sizes[comm]} nodes)')
    )

ax.legend(handles=legend_elements, 
         loc='upper right', 
         fontsize=12,
         title='Top 10 Communities',
         title_fontsize=14)

ax.set_title('Facebook Network - Community Structure Visualization\n' + 
            f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, ' +
            f'{num_communities} communities detected',
            fontsize=18, fontweight='bold', pad=20)
ax.axis('off')
plt.tight_layout()

print("  Saving full network visualization...")
plt.savefig('results/visualizations/network_full_communities.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: results/visualizations/network_full_communities.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: ZOOMED VIEW OF LARGEST COMMUNITY
# ============================================================================

print("\n[STEP 3] Creating zoomed view of largest community...")

largest_community = community_sizes.idxmax()
nodes_in_largest = [node for node, comm in communities_dict.items() 
                   if comm == largest_community]

# Create subgraph of largest community + its neighbors
extended_nodes = set(nodes_in_largest)
for node in nodes_in_largest:
    extended_nodes.update(G.neighbors(node))

G_largest = G.subgraph(extended_nodes)
print(f"  Largest community: {len(nodes_in_largest)} core nodes")
print(f"  Extended subgraph: {G_largest.number_of_nodes()} nodes")

# Layout for subgraph
pos_sub = nx.spring_layout(G_largest, k=0.5, iterations=50, seed=42)

# Color nodes
node_colors_sub = []
for node in G_largest.nodes():
    comm = communities_dict.get(node, -1)
    if comm == largest_community:
        # Core community nodes in bright color
        color_idx = unique_communities.index(comm) % len(colors_palette)
        node_colors_sub.append(colors_palette[color_idx])
    else:
        # Neighboring nodes in light gray
        node_colors_sub.append((0.8, 0.8, 0.8, 0.6))

# Node sizes
degrees_sub = dict(G_largest.degree())
node_sizes_sub = [degrees_sub[node] * 10 for node in G_largest.nodes()]

fig, ax = plt.subplots(figsize=(16, 16), facecolor='white')

# Draw
nx.draw_networkx_edges(G_largest, pos_sub, alpha=0.2, width=1, 
                      edge_color='gray', ax=ax)
nx.draw_networkx_nodes(G_largest, pos_sub,
                      node_color=node_colors_sub,
                      node_size=node_sizes_sub,
                      alpha=0.9,
                      edgecolors='black',
                      linewidths=0.5,
                      ax=ax)

ax.set_title(f'Largest Community (Community {largest_community})\n' +
            f'{len(nodes_in_largest)} core members + connections to neighbors',
            fontsize=16, fontweight='bold', pad=20)
ax.axis('off')
plt.tight_layout()

plt.savefig('results/visualizations/network_largest_community_zoom.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: results/visualizations/network_largest_community_zoom.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: TOP 5 COMMUNITIES SIDE-BY-SIDE
# ============================================================================

print("\n[STEP 4] Creating top 5 communities comparison...")

top_5_communities = community_sizes.head(5).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(20, 14), facecolor='white')
axes = axes.flatten()

for idx, comm in enumerate(top_5_communities):
    ax = axes[idx]

    # Get nodes in this community
    nodes_in_comm = [node for node, c in communities_dict.items() if c == comm]

    # Create subgraph
    G_comm = G.subgraph(nodes_in_comm)

    # Layout
    pos_comm = nx.spring_layout(G_comm, k=0.8, iterations=50, seed=42)

    # Draw
    nx.draw_networkx_edges(G_comm, pos_comm, alpha=0.3, width=0.5, ax=ax)

    # Get node colors for this community
    color_idx = unique_communities.index(comm) % len(colors_palette)

    degrees_comm = dict(G_comm.degree())
    node_sizes_comm = [degrees_comm[node] * 15 for node in G_comm.nodes()]

    nx.draw_networkx_nodes(G_comm, pos_comm,
                          node_color=[colors_palette[color_idx]] * len(G_comm.nodes()),
                          node_size=node_sizes_comm,
                          alpha=0.8,
                          edgecolors='black',
                          linewidths=0.5,
                          ax=ax)

    # Statistics
    density = nx.density(G_comm)
    avg_degree = sum(degrees_comm.values()) / len(degrees_comm) if degrees_comm else 0

    ax.set_title(f'Community {comm}\n{len(nodes_in_comm)} nodes, ' +
                f'Density={density:.3f}, Avg Degree={avg_degree:.1f}',
                fontsize=12, fontweight='bold')
    ax.axis('off')

# Remove extra subplot
axes[-1].axis('off')

plt.suptitle('Top 5 Largest Communities - Internal Structure',
            fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()

plt.savefig('results/visualizations/network_top5_communities.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: results/visualizations/network_top5_communities.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: CIRCULAR LAYOUT (PRETTIER FOR REPORTS)
# ============================================================================

print("\n[STEP 5] Creating circular layout visualization...")

# Sample 500 nodes for cleaner circular view
np.random.seed(42)
sampled_nodes = np.random.choice(list(G.nodes()), size=min(500, G.number_of_nodes()), replace=False)
G_sample = G.subgraph(sampled_nodes)

# Circular layout
pos_circular = nx.circular_layout(G_sample)

# Node colors
node_colors_circular = []
for node in G_sample.nodes():
    comm = communities_dict.get(node, 0)
    color_idx = unique_communities.index(comm) % len(colors_palette)
    node_colors_circular.append(colors_palette[color_idx])

# Node sizes
degrees_circular = dict(G_sample.degree())
node_sizes_circular = [degrees_circular[node] * 15 for node in G_sample.nodes()]

fig, ax = plt.subplots(figsize=(18, 18), facecolor='white')

# Draw with circular layout
nx.draw_networkx_edges(G_sample, pos_circular, alpha=0.15, width=0.5,
                      edge_color='gray', ax=ax)
nx.draw_networkx_nodes(G_sample, pos_circular,
                      node_color=node_colors_circular,
                      node_size=node_sizes_circular,
                      alpha=0.85,
                      edgecolors='black',
                      linewidths=0.5,
                      ax=ax)

ax.set_title(f'Facebook Network - Circular Layout (Sample of {len(G_sample.nodes())} nodes)\n' +
            'Node colors represent different communities',
            fontsize=16, fontweight='bold', pad=20)
ax.axis('off')
plt.tight_layout()

plt.savefig('results/visualizations/network_circular_communities.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: results/visualizations/network_circular_communities.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUCCESS! ENHANCED VISUALIZATIONS COMPLETE!")
print("="*80)
print("\nGenerated 4 network visualizations:")
print("  1. network_full_communities.png (ALL 4,039 nodes)")
print("  2. network_largest_community_zoom.png (Largest community zoomed)")
print("  3. network_top5_communities.png (Top 5 communities comparison)")
print("  4. network_circular_communities.png (Circular layout, 500 nodes sample)")
print("\nAll saved in: results/visualizations/")
print("="*80)
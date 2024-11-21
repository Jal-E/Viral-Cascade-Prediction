import networkx as nx
import matplotlib.pyplot as plt
from community import best_partition
from collections import Counter

# Assuming `G` is already constructed from your Network_creation.py
# Replace the following line with your actual network:
from Network_creation import construct_network
from Dataset_exploration import df

G = construct_network(df)

# 1. Basic Network Statistics
def basic_network_stats(G):
    print(f"Number of Nodes: {G.number_of_nodes()}")
    print(f"Number of Edges: {G.number_of_edges()}")
    print(f"Average Degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"Density: {nx.density(G):.4f}")

# 2. Degree Distribution
def plot_degree_distribution(G):
    degrees = [d for _, d in G.degree()]
    plt.hist(degrees, bins=30, color='blue', alpha=0.7)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

# 3. Top Connected Nodes
def top_connected_nodes(G, top_n=5):
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_n]
    print(f"Top {top_n} nodes by degree:")
    for node, degree in top_nodes:
        print(f"Node: {node}, Degree: {degree}")

# 4. Community Detection
def community_detection(G):
    partition = best_partition(G.to_undirected())
    num_communities = len(set(partition.values()))
    print(f"Number of communities detected: {num_communities}")

    community_counts = Counter(partition.values())
    print("Community sizes:", community_counts)

# 5. Largest Connected Component
def largest_connected_component(G):
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)
    print(f"Largest strongly connected component has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
    return subgraph

# 6. Sample Edge Weights
def sample_edge_weights(G, sample_size=5):
    print("Sample edge weights:")
    for u, v, d in list(G.edges(data=True))[:sample_size]:
        print(f"Edge ({u} -> {v}) with weight {d['weight']}")

# 7. Visualize Subgraph
def visualize_subgraph(G, num_nodes=50):
    # Reduce to a subgraph if the network is too large
    sub_nodes = list(G.nodes())[:num_nodes]
    subgraph = G.subgraph(sub_nodes)
    print(f"Visualizing a subgraph with {len(subgraph)} nodes and {subgraph.number_of_edges()} edges.")

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subgraph)  # Positioning nodes for visualization
    nx.draw(
        subgraph, pos,
        with_labels=False,
        node_size=50,
        node_color="blue",
        edge_color="gray",
        alpha=0.7
    )
    plt.title("Subgraph Visualization")
    plt.show()

# Run all investigations
if __name__ == "__main__":
    print("### Basic Network Statistics ###")
    basic_network_stats(G)
    
    print("\n### Degree Distribution ###")
    plot_degree_distribution(G)

    print("\n### Top Connected Nodes ###")
    top_connected_nodes(G, top_n=5)

    print("\n### Community Detection ###")
    community_detection(G)

    print("\n### Largest Connected Component ###")
    largest_cc = largest_connected_component(G)

    print("\n### Sample Edge Weights ###")
    sample_edge_weights(G, sample_size=5)

    print("\n### Visualize Subgraph ###")
    visualize_subgraph(G, num_nodes=50)

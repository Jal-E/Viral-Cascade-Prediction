import networkx as nx
import matplotlib.pyplot as plt
from community import best_partition
from collections import Counter

# 1. Basic Network Statistics
def basic_statistics(G):
    print("### Basic Network Statistics ###")
    print(f"Number of Nodes: {G.number_of_nodes()}")
    print(f"Number of Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")
    print(f"Is Directed: {nx.is_directed(G)}")
    if nx.is_directed(G):
        print(f"Strongly Connected Components: {nx.number_strongly_connected_components(G)}")
    else:
        print(f"Connected Components: {nx.number_connected_components(G)}")

# 2. Degree Distribution
def plot_degree_distribution(G):
    print("\n### Degree Distribution ###")
    degrees = [d for _, d in G.degree()]
    plt.hist(degrees, bins=30, color='blue', alpha=0.7)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

# 3. Community Analysis
def community_analysis(G):
    print("\n### Community Analysis ###")
    partition = best_partition(G.to_undirected())  # Community detection using Louvain
    num_communities = len(set(partition.values()))
    print(f"Number of Communities: {num_communities}")

    # Analyze community sizes
    community_counts = Counter(partition.values())
    print(f"Community Sizes: {dict(community_counts)}")

    # Plot community size distribution
    plt.bar(community_counts.keys(), community_counts.values(), color='green', alpha=0.7)
    plt.title("Community Size Distribution")
    plt.xlabel("Community ID")
    plt.ylabel("Number of Nodes")
    plt.show()

# 4. Temporal Analysis
def temporal_analysis(df):
    print("\n### Temporal Analysis ###")
    print(f"Time Range: {df['dateadded_post'].min()} to {df['dateadded_post'].max()}")

    # Plot cascades over time
    df['year_month'] = df['dateadded_post'].dt.to_period('M')  # Group by month
    cascade_counts = df.groupby('year_month').size()
    cascade_counts.plot(kind='bar', figsize=(15, 6), color='orange', alpha=0.7)
    plt.title("Cascade Activity Over Time")
    plt.xlabel("Year-Month")
    plt.ylabel("Number of Cascades")
    plt.show()

# 5. Clustering Coefficient
def clustering_analysis(G):
    print("\n### Clustering Coefficient Analysis ###")
    avg_clustering = nx.average_clustering(G.to_undirected())
    print(f"Average Clustering Coefficient: {avg_clustering:.4f}")

    # Distribution of clustering coefficients
    clustering_coeffs = nx.clustering(G.to_undirected()).values()
    plt.hist(list(clustering_coeffs), bins=30, color='purple', alpha=0.7)
    plt.title("Clustering Coefficient Distribution")
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Frequency")
    plt.show()

# Run all network investigations
def run_network_investigation(G, df):
    basic_statistics(G)
    plot_degree_distribution(G)
    community_analysis(G)
    temporal_analysis(df)
    clustering_analysis(G)

# Assuming `G` is your network and `df` is your dataset
if __name__ == "__main__":
    from Network_creation import construct_network
    from Dataset_exploration import df  # Dataset containing timestamps and other information

    # Construct the network
    G = construct_network(df)

    # Run network investigations
    run_network_investigation(G, df)

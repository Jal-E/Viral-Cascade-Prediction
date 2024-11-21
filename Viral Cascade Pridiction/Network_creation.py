import networkx as nx
from community import best_partition
from Dataset_exploration import df  # Import the dataset

# 1. Network Creation: Build Directed Graph
def construct_network(df):
    """
    Constructs a directed graph where nodes are users and edges represent repost relationships.
    """
    G = nx.DiGraph()
    for topic_id, group in df.groupby('topic_id'):
        group = group.sort_values(by='dateadded_post')
        users = group['user_id'].to_numpy()
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                if group.iloc[j]['dateadded_post'] >= group.iloc[i]['dateadded_post']:
                    if G.has_edge(users[i], users[j]):
                        G[users[i]][users[j]]['weight'] += 1
                    else:
                        G.add_edge(users[i], users[j], weight=1)
    return G

# 2. Community Detection
def detect_communities(G):
    """
    Performs community detection using the Louvain algorithm.
    """
    partition = best_partition(G.to_undirected())  # Louvain algorithm
    num_communities = len(set(partition.values()))
    print(f"Detected {num_communities} communities.")
    return partition

# 3. Cascade Extraction
def extract_cascades(G, df):
    """
    Extract cascades as subsets of nodes and edges based on topics in the dataset.
    """
    cascades = []
    for topic_id, group in df.groupby('topic_id'):
        cascade_nodes = list(group['user_id'])
        cascade_edges = [(u, v) for u, v in G.edges() if u in cascade_nodes and v in cascade_nodes]
        cascades.append((cascade_nodes, cascade_edges))
    print(f"Extracted {len(cascades)} cascades.")
    return cascades

# 4. Frontier Definitions
def calculate_frontiers(G, adopters, timestamps, lambda_time):
    """
    Calculate frontiers, λ frontiers, and non-adopters for a cascade.
    """
    frontiers = {neighbor for adopter in adopters for neighbor in G.successors(adopter) if neighbor not in adopters}
    lambda_frontiers = {
        node for node in frontiers
        if min(
            (timestamps[node] - timestamps[adopter]).total_seconds() / 3600
            for adopter in adopters if adopter in timestamps and node in timestamps
        ) <= lambda_time
    }
    non_adopters = frontiers - lambda_frontiers
    return frontiers, lambda_frontiers, non_adopters

# 5. Full Pipeline
if __name__ == "__main__":
    print("Using dataset imported from `Dataset_exploration`.")
    
    # Step 1: Construct Network
    G = construct_network(df)
    print(f"Constructed network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Step 2: Community Detection
    partition = detect_communities(G)

    # Step 3: Cascade Extraction
    cascades = extract_cascades(G, df)

    # Step 4: Frontier Definitions for a Single Cascade (Example)
    adopters = set(df['user_id'][:10])  # Example: First 10 adopters
    timestamps = {row['user_id']: row['dateadded_post'] for _, row in df.iterrows()}
    lambda_time = 14  # Example: λ = 14 hours

    frontiers, lambda_frontiers, non_adopters = calculate_frontiers(G, adopters, timestamps, lambda_time)
    print(f"Frontiers: {len(frontiers)}")
    print(f"λ Frontiers: {len(lambda_frontiers)}")
    print(f"Non-Adopters: {len(non_adopters)}")

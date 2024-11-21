import pandas as pd 
from collections import Counter
from Network_creation import construct_network, detect_communities, calculate_frontiers

# Gini Impurity Calculation
def calculate_gini_impurity(nodes, partition):
    """
    Calculate Gini impurity for nodes based on their community distribution.
    """
    community_counts = Counter(partition[node] for node in nodes if node in partition)
    total = sum(community_counts.values())
    if total == 0:
        return 0  # Avoid division by zero
    return 1 - sum((count / total) ** 2 for count in community_counts.values())

# Feature Extraction Function
def extract_features(G, adopters, lambda_frontiers, lambda_non_adopters, partition, timestamps, topic_id):
    """
    Extract features for a single cascade based on its structural diversity.
    Adds topic_id to the feature set.
    """
    num_adopters = len(adopters)
    
    # Edge case handling
    if num_adopters == 0:
        return {
            'topic_id': topic_id,  # Add topic_id as the first feature
            'avg_time_to_adoption': 0,
            
        }

    # Origin time (t0) is the earliest adoption time (the origin time of the cascade)
    origin_time = min(timestamps[node] for node in adopters if node in timestamps)

    # Calculate the relative adoption times (t_i - t0)
    adoption_times = [
        (timestamps[node] - origin_time).total_seconds() for node in adopters if node in timestamps
    ]

    # Apply the formula: sum(t_i^θ) / m
    avg_time_to_adoption = sum(adoption_times) / num_adopters if num_adopters > 0 else 0

    features = {
        'topic_id': topic_id,  # Add topic_id as the first feature
        'num_communities_adopters': len(set(partition[node] for node in adopters if node in partition)),
        'num_communities_lambda_frontiers': len(set(partition[node] for node in lambda_frontiers if node in partition)),
        'num_communities_lambda_non_adopters': len(set(partition[node] for node in lambda_non_adopters if node in partition)),

        'gini_impurity_adopters': calculate_gini_impurity(adopters, partition),
        'gini_impurity_lambda_frontiers': calculate_gini_impurity(lambda_frontiers, partition),
        'gini_impurity_lambda_non_adopters': calculate_gini_impurity(lambda_non_adopters, partition),

        'overlap_adopters_lambda_frontiers': len(set(adopters) & set(lambda_frontiers)),
        'overlap_adopters_lambda_non_adopters': len(set(adopters) & set(lambda_non_adopters)),
        'overlap_lambda_frontiers_lambda_non_adopters': len(set(lambda_frontiers) & set(lambda_non_adopters)),

        # The 10th feature: average time to adoption in hours
        'avg_time_to_adoption': avg_time_to_adoption / 3600  # Convert to hours
    }
    return features

# Process Cascades with Virality Calculation
def process_cascades(G, df, partition, lambda_time=14, alpha=10):
    """
    Process all cascades in the dataset and extract features.
    Adds a 'virality' column to indicate if the cascade is viral or not.
    """
    all_features = []
    timestamps = {row['user_id']: row['dateadded_post'] for _, row in df.iterrows()}

    for topic_id, group in df.groupby('topic_id'):
        adopters = set(group['user_id'])
        adopters = {node for node in adopters if node in G}  # Keep only nodes in G
        if not adopters:
            continue

        frontiers, lambda_frontiers, lambda_non_adopters = calculate_frontiers(G, adopters, timestamps, lambda_time)
        features = extract_features(G, adopters, lambda_frontiers, lambda_non_adopters, partition, timestamps, topic_id)

        cascade_size = len(adopters)  # Total adopters in the cascade
        virality_threshold = 4 * alpha
        features['virality'] = 1 if cascade_size >= virality_threshold else 0

        all_features.append(features)

    return pd.DataFrame(all_features)

# Main Pipeline for Feature Extraction
if __name__ == "__main__":
    from Dataset_exploration import df  # Import the dataset

    print("Using dataset imported from `Dataset_exploration`.")

    G = construct_network(df)
    print(f"Constructed network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    partition = detect_communities(G)

    lambda_time = 14  # λ = 14 hours
    alpha = 10  # Alpha value for virality threshold
    cascade_features = process_cascades(G, df, partition, lambda_time, alpha)
    print("Feature Extraction Complete.")

    cascade_features.to_csv("cascade_features.csv", index=False)
    print("Features saved to `cascade_features.csv`.")

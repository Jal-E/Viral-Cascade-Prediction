import pandas as pd
from collections import Counter
from Network_creation import (
    construct_network_for_topic,
    calculate_frontiers_for_topic,
    save_net,
    visualize_network,
    load_and_filter_data,
    calculate_alpha_times,
    get_roots
)
import community as community_louvain  # Import the community module

# Gini Impurity Calculation
def calculate_gini_impurity(nodes, partition):
    """
    Calculate Gini impurity based on community sizes.
    """
    community_sizes = {}
    for node in nodes:
        if node in partition:
            comm = partition[node]
            community_sizes[comm] = community_sizes.get(comm, 0) + 1

    sizes = list(community_sizes.values())
    total = sum(sizes)
    if total == 0:
        return 0  
    return 1 - sum((size / total) ** 2 for size in sizes)

# Overlap/Shared Communities Calculation
def calculate_shared_communities(partition, group1, group2):
    """
    Calculate the number of shared communities between two groups.
    """
    communities1 = set(partition[node] for node in group1 if node in partition)
    communities2 = set(partition[node] for node in group2 if node in partition)
    return len(communities1 & communities2)

# Average Time to Adoption Calculation
def calculate_avg_time_to_adoption(timestamps, adopters):
    """
    Calculate average time to adoption based on consecutive adopter times.
    """
    adoption_times = sorted(timestamps[node] for node in adopters if node in timestamps)
    if len(adoption_times) < 2:
        return 0  # Not enough data for average
    total_time = sum(
        (adoption_times[i + 1] - adoption_times[i]).total_seconds()
        for i in range(len(adoption_times) - 1)
    )
    return total_time / (len(adoption_times) - 1)

# Feature Extraction Function
def extract_features(G, adopters, lambda_frontiers, lambda_non_adopters, partition, timestamps, topic_id, num_adopters, alpha):
    """
    Extract features for a single cascade.
    """
    # Determine if the topic is viral
    is_viral = int(num_adopters >= 4 * alpha)  # 1 if viral, 0 otherwise
    
    # Edge case handling: No adopters
    if num_adopters == 0:
        return {
            'topic_id': topic_id,  
            'gini_impurity_adopters': 0,
            'gini_impurity_lambda_frontiers': 0,
            'gini_impurity_lambda_non_adopters': 0,
            'num_communities_adopters': 0,
            'num_communities_lambda_frontiers': 0,
            'num_communities_lambda_non_adopters': 0,
            'overlap_adopters_lambda_frontiers': 0,
            'overlap_adopters_lambda_non_adopters': 0,
            'overlap_lambda_frontiers_lambda_non_adopters': 0,
            'avg_time_to_adoption': 0,
            'virality': is_viral
        }

    # Calculate adoption times relative to origin (t0)
    avg_time_to_adoption = calculate_avg_time_to_adoption(timestamps, adopters)

    # Extract features
    features = {
        'topic_id': topic_id,
        # Gini impurity
        'gini_impurity_adopters': calculate_gini_impurity(adopters, partition),
        'gini_impurity_lambda_frontiers': calculate_gini_impurity(lambda_frontiers, partition),
        'gini_impurity_lambda_non_adopters': calculate_gini_impurity(lambda_non_adopters, partition),

        # Number of communities
        'num_communities_adopters': len(set(partition[node] for node in adopters if node in partition)),
        'num_communities_lambda_frontiers': len(set(partition[node] for node in lambda_frontiers if node in partition)),
        'num_communities_lambda_non_adopters': len(set(partition[node] for node in lambda_non_adopters if node in partition)),

        # Overlaps
        'overlap_adopters_lambda_frontiers': calculate_shared_communities(partition, adopters, lambda_frontiers),
        'overlap_adopters_lambda_non_adopters': calculate_shared_communities(partition, adopters, lambda_non_adopters),
        'overlap_lambda_frontiers_lambda_non_adopters': calculate_shared_communities(partition, lambda_frontiers, lambda_non_adopters),

        # Average time to adoption (hours)
        'avg_time_to_adoption': avg_time_to_adoption / 3600,  # Convert to hours

        # Virality
        'virality': is_viral
    }
    return features

# Process Cascades and Extract Features
def process_cascades(df, alpha, lambda_time):
    """
    Process all cascades in the dataset, construct networks, and extract features.
    """
    all_features = []
    alpha_times = calculate_alpha_times(df, alpha)  # Calculate alpha times
    roots = get_roots(df)  # Identify root users

    for topic_id, group in df.groupby('topic_id'):
        alpha_time = alpha_times.get(topic_id)
        if alpha_time is None:
            continue

        # Filter group up to alpha_time
        filtered_group = group[group['dateadded_post'] <= alpha_time]

        # Construct network for the topic
        G = construct_network_for_topic(filtered_group, alpha_time)  # Pass alpha_time explicitly
        save_net(G, topic_id)  # Save network for inspection
        visualize_network(G, topic_id)  # Visualize network

        # Get all adopters and their timestamps
        all_adopters = set(filtered_group['user_id'].unique())  # All unique adopters in the cascade
        adopters = set(filtered_group['user_id'][:alpha])  # Early adopters for the cascade
        num_adopters = len(all_adopters)  # Total number of adopters for the cascade
        timestamps = {row['user_id']: row['dateadded_post'] for _, row in filtered_group.iterrows()}

        # Debugging: Print num_adopters and virality threshold
        print(f"Topic: {topic_id}, Num Adopters: {num_adopters}, Viral Threshold: {4 * alpha}")

        # Calculate frontiers
        frontiers, lambda_frontiers, lambda_non_adopters = calculate_frontiers_for_topic(G, adopters, timestamps, lambda_time)

        # Detect communities using Louvain method
        partition = community_louvain.best_partition(G.to_undirected())

        # Extract features
        features = extract_features(G, adopters, lambda_frontiers, lambda_non_adopters, partition, timestamps, topic_id, num_adopters, alpha)
        all_features.append(features)

    return pd.DataFrame(all_features)

# Main Pipeline for Feature Extraction
if __name__ == "__main__":
    # Load dataset
    forum_id = 4
    min_post_length = 10
    min_posts_per_user = 5
    min_threads_per_user = 2
    classification_threshold = 0.5

    # Load and preprocess data
    df = load_and_filter_data(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold)

    # Parameters
    alpha = 10  # Number of adopters for defining cascades
    lambda_time = 14  # λ in hours

    # Process cascades and extract features
    cascade_features = process_cascades(df, alpha, lambda_time)
    print("Feature Extraction Complete.")

    # Save features to CSV
    cascade_features.to_csv("cascade_features.csv", index=False)
    print("Features saved to `cascade_features.csv`.")

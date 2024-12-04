import pandas as pd
from collections import Counter
from Network_creation import (
    construct_network_for_topic,
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
    community_sizes = Counter(partition[node] for node in nodes if node in partition)
    total = sum(community_sizes.values())
    if total == 0:
        return 0
    return 1 - sum((size / total) ** 2 for size in community_sizes.values())

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

def extract_features(G, adopters, lambda_frontiers, lambda_non_adopters, partition, timestamps, topic_id, num_adopters, alpha):
    """
    Extract features for a single cascade.
    """
    is_viral = int(num_adopters >= 4 * alpha)  # 1 if viral, 0 otherwise
    avg_time_to_adoption = calculate_avg_time_to_adoption(timestamps, adopters)
    features = {
        'topic_id': topic_id,
        'gini_impurity_adopters': calculate_gini_impurity(adopters, partition),
        'gini_impurity_lambda_frontiers': calculate_gini_impurity(lambda_frontiers, partition),
        'gini_impurity_lambda_non_adopters': calculate_gini_impurity(lambda_non_adopters, partition),
        'num_communities_adopters': len(set(partition[node] for node in adopters if node in partition)),
        'num_communities_lambda_frontiers': len(set(partition[node] for node in lambda_frontiers if node in partition)),
        'num_communities_lambda_non_adopters': len(set(partition[node] for node in lambda_non_adopters if node in partition)),
        'overlap_adopters_lambda_frontiers': calculate_shared_communities(partition, adopters, lambda_frontiers),
        'overlap_adopters_lambda_non_adopters': calculate_shared_communities(partition, adopters, lambda_non_adopters),
        'overlap_lambda_frontiers_lambda_non_adopters': calculate_shared_communities(partition, lambda_frontiers, lambda_non_adopters),
        'avg_time_to_adoption': avg_time_to_adoption / 3600,  # Convert to hours
        'virality': is_viral
    }
    return features

def process_cascades(df, alpha, lambda_time):
    """
    Process all cascades in the dataset, construct networks, and extract features.
    """
    all_features = []
    alpha_times = calculate_alpha_times(df, alpha)
    roots = get_roots(df)

    for topic_id, group in df.groupby('topic_id'):
        alpha_time = alpha_times.get(topic_id)
        if not alpha_time:
            continue

        # Construct the network and determine adopters
        G, adopters = construct_network_for_topic(group, alpha_time, alpha)
        save_net(G, topic_id)
        visualize_network(G, topic_id)

        # Extract timestamps
        timestamps = {row['user_id']: row['dateadded_post'] for _, row in group.iterrows()}
        num_adopters = len(set(group['user_id']))

        # Calculate frontiers, lambda frontiers, and non-adopters
        frontiers, lambda_frontiers, lambda_non_adopters = calculate_frontiers_for_topic(G, adopters, timestamps, lambda_time)

        # Debugging: Display information for verification
        print(f"Processing topic_id: {topic_id}")
        print(f"Adopters: {adopters}")
        print(f"Frontiers: {frontiers}")
        print(f"Lambda Frontiers: {lambda_frontiers}")
        print(f"Lambda Non-Adopters: {lambda_non_adopters}")

        if not frontiers:
            print(f"No valid frontiers for topic_id {topic_id}. Skipping.")
            continue

        # Detect communities using Louvain method
        partition = community_louvain.best_partition(G.to_undirected())

        # Extract features
        features = extract_features(G, adopters, lambda_frontiers, lambda_non_adopters, partition, timestamps, topic_id, num_adopters, alpha)
        all_features.append(features)

    return pd.DataFrame(all_features)

# Main Pipeline
if __name__ == "__main__":
    forum_id = 4
    min_post_length = 10
    min_posts_per_user = 5
    min_threads_per_user = 2
    classification_threshold = 0.5

    df = load_and_filter_data(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold)
    alpha = 10
    lambda_time = 24

    cascade_features = process_cascades(df, alpha, lambda_time)
    print("Feature Extraction Complete.")

    cascade_features.to_csv("cascade_features.csv", index=False)
    print("Features saved to `cascade_features.csv`.")

import os
import pandas as pd
import networkx as nx
from sqlalchemy import create_engine
from collections import Counter
import community as community_louvain
import numpy as np

# Directories for outputs
NETWORK_DIR = "networks"
VISUALIZATION_DIR = "visualizations"
os.makedirs(NETWORK_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def get_db_connection():
    engine = create_engine('postgresql://postgres:1234@localhost:5432/Oct')
    return engine.connect()

def load_and_filter_data(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold):
    conn = get_db_connection()
    query = f"""
    SELECT posts.user_id, posts.topic_id, posts.post_id, posts.dateadded_post, 
           LENGTH(posts.content_post) AS post_length, topics.classification2_topic
    FROM posts
    INNER JOIN topics ON posts.topic_id = topics.topic_id
    WHERE topics.forum_id = %s 
          AND topics.classification2_topic >= %s 
          AND LENGTH(posts.content_post) >= %s
    """
    df = pd.read_sql(query, conn, params=(forum_id, classification_threshold, min_post_length))
    conn.close()
    
    df['dateadded_post'] = pd.to_datetime(df['dateadded_post'], utc=True)
    # Filter users by post counts
    eligible_users = df['user_id'].value_counts()[lambda x: x >= min_posts_per_user].index
    df = df[df['user_id'].isin(eligible_users)]
    # Filter users by threads
    eligible_thread_users = df.groupby('user_id')['topic_id'].nunique()[lambda x: x >= min_threads_per_user].index
    df = df[df['user_id'].isin(eligible_thread_users)]

    return df

def calculate_gini_impurity(nodes, partition):
    community_sizes = Counter(partition[node] for node in nodes if node in partition)
    total = sum(community_sizes.values())
    if total == 0:
        return 0
    return 1 - sum((size / total) ** 2 for size in community_sizes.values())

def calculate_shared_communities(partition, group1, group2):
    communities1 = set(partition[node] for node in group1 if node in partition)
    communities2 = set(partition[node] for node in group2 if node in partition)
    return len(communities1 & communities2)

def calculate_avg_time_to_adoption(timestamps, adopters):
    adoption_times = sorted(timestamps[node] for node in adopters if node in timestamps)
    if len(adoption_times) < 2:
        return 0
    total_time = sum(
        (adoption_times[i + 1] - adoption_times[i]).total_seconds()
        for i in range(len(adoption_times) - 1)
    )
    return total_time / (len(adoption_times) - 1)

def extract_features(G, adopters, lambda_frontiers, lambda_non_adopters, partition, timestamps, topic_id, group, alpha, beta):
    # Determine virality: topic is viral if it reaches at least beta posts
    is_viral = int(len(group) >= beta)
    
    # Compute delta_t if we have at least beta posts
    group_sorted = group.sort_values(by='dateadded_post')
    if len(group_sorted) >= beta:
        alpha_post_time = group_sorted.iloc[alpha-1]['dateadded_post']
        beta_post_time = group_sorted.iloc[beta-1]['dateadded_post']
        delta_t_seconds = (beta_post_time - alpha_post_time).total_seconds()
        delta_t_hours = delta_t_seconds / 3600.0
    else:
        delta_t_hours = np.nan

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
        'avg_time_to_adoption': avg_time_to_adoption / 3600.0,  # converting to hours
        'virality': is_viral,
        'delta_t': delta_t_hours
    }
    return features

def process_all_topics_with_global_network(df, alpha, lambda_time, beta):
    lambda_time_delta = pd.Timedelta(hours=lambda_time)

    # Create a global graph
    global_G = nx.DiGraph()
    for topic_id, group in df.groupby('topic_id'):
        group_sorted = group.sort_values(by='dateadded_post')
        users_in_topic = group_sorted['user_id'].tolist()
        times_in_topic = group_sorted['dateadded_post'].tolist()
        for i in range(len(users_in_topic)):
            for j in range(i+1, len(users_in_topic)):
                u = users_in_topic[i]
                v = users_in_topic[j]
                global_G.add_edge(u, v, formed_at=times_in_topic[j])

    # Identify cascade topics
    cascade_topics = [t_id for t_id, grp in df.groupby('topic_id') if len(grp) >= alpha]

    results = []
    for topic_id in cascade_topics:
        group = df[df['topic_id'] == topic_id].sort_values(by='dateadded_post')
        if len(group) < alpha:
            continue

        # Determine adopters
        adopters = group['user_id'].iloc[:alpha].tolist()
        alpha_times = group['dateadded_post'].iloc[:alpha].tolist()
        adopter_to_time = dict(zip(adopters, alpha_times))

        # Determine frontiers, lambda_frontiers, non_adopters
        frontier_exposures = {}
        for adopter in adopters:
            a_time = adopter_to_time[adopter]
            if adopter in global_G:
                for successor in global_G.successors(adopter):
                    edge_data = global_G.get_edge_data(adopter, successor)
                    formed_at = edge_data['formed_at']
                    if formed_at <= a_time:
                        if successor not in frontier_exposures or frontier_exposures[successor] > a_time:
                            frontier_exposures[successor] = a_time

        frontiers = set(frontier_exposures.keys())
        lambda_frontiers = set()
        non_adopters = set()

        timestamps = {row['user_id']: row['dateadded_post'] for _, row in group.iterrows()}
        for frontier in frontiers:
            exposure_time = frontier_exposures[frontier]
            f_post = group[group['user_id'] == frontier]
            if not f_post.empty:
                f_time = f_post['dateadded_post'].min()
                if f_time <= exposure_time + lambda_time_delta:
                    lambda_frontiers.add(frontier)
                else:
                    non_adopters.add(frontier)
            else:
                non_adopters.add(frontier)

        # Run community detection on global graph (undirected)
        partition = community_louvain.best_partition(global_G.to_undirected())

        features = extract_features(global_G, adopters, lambda_frontiers, non_adopters, partition, timestamps, topic_id, group, alpha, beta)
        results.append(features)

    return pd.DataFrame(results)

if __name__ == "__main__":
    forum_id = 4
    min_post_length = 10
    min_posts_per_user = 5
    min_threads_per_user = 2
    classification_threshold = 0.5
    lambda_time = 24
    alpha = 10
    beta = 40  # For virality definition

    df = load_and_filter_data(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold)
    cascade_features = process_all_topics_with_global_network(df, alpha, lambda_time, beta)
    print("Feature Extraction Complete:")
    print(cascade_features)
    cascade_features.to_csv("cascade_features.csv", index=False)
    print("Features saved to `cascade_features.csv`.")

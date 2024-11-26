import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle as pk
from sqlalchemy import create_engine
from community import best_partition

# Create directories for outputs
NETWORK_DIR = "networks"
VISUALIZATION_DIR = "visualizations"
os.makedirs(NETWORK_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Database connection
def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    """
    engine = create_engine('postgresql://postgres:1234@localhost:5432/Oct')
    return engine.connect()

# Load and filter data
def load_and_filter_data(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold):
    """
    Load data from the database and filter it based on post length, user activity, and thread activity.
    """
    conn = get_db_connection()

    # SQL query with filtering logic incorporated
    query = f"""
    SELECT posts.user_id, posts.topic_id, posts.post_id, posts.dateadded_post, 
           LENGTH(posts.content_post) AS post_length, topics.classification2_topic
    FROM posts
    INNER JOIN topics ON posts.topic_id = topics.topic_id
    WHERE topics.forum_id = %s 
          AND topics.classification2_topic >= %s 
          AND LENGTH(posts.content_post) >= %s
    """

    # Execute query with parameters
    df = pd.read_sql(query, conn, params=(forum_id, classification_threshold, min_post_length))
    conn.close()

    # Convert to datetime
    df['dateadded_post'] = pd.to_datetime(df['dateadded_post'], utc=True)

    # Filter users by minimum posts
    user_post_counts = df['user_id'].value_counts()
    eligible_users = user_post_counts[user_post_counts >= min_posts_per_user].index
    df = df[df['user_id'].isin(eligible_users)]

    # Filter users by threads participated
    user_thread_counts = df.groupby('user_id')['topic_id'].nunique()
    eligible_thread_users = user_thread_counts[user_thread_counts >= min_threads_per_user].index
    df = df[df['user_id'].isin(eligible_thread_users)]

    return df


# Identify roots for each topic
def get_roots(df):
    """
    Find the root user (first poster) for each topic.
    """
    roots = {}
    for topic_id, group in df.groupby('topic_id'):
        group = group.sort_values(by='dateadded_post')
        roots[topic_id] = group.iloc[0]['user_id']
    return roots

# Save network to a file
def save_net(G, topic_id):
    """
    Save the network graph to a pickle file.
    """
    filename = os.path.join(NETWORK_DIR, f"network_topic_{topic_id}.pkl")
    with open(filename, 'wb') as f:
        pk.dump(G, f)
    print(f"Network saved to {filename}")

# Visualize network
def visualize_network(G, topic_id):
    """
    Visualize the network graph and save it to a file.
    """
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)  # Spring layout for visualization
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    plt.title(f"Network Visualization for Topic {topic_id}")

    filename = os.path.join(VISUALIZATION_DIR, f"network_topic_{topic_id}.png")
    plt.savefig(filename, dpi=300)
    plt.close()  # Close the plot to prevent popping up
    print(f"Network visualization saved to {filename}")

# 1. Calculate alpha_time for each topic
def calculate_alpha_times(df, alpha):
    """
    Calculate alpha_time for each topic as the timestamp of the alpha-th adopter.
    """
    alpha_times = {}
    for topic_id, group in df.groupby('topic_id'):
        group = group.sort_values(by='dateadded_post')
        if len(group) >= alpha:
            alpha_times[topic_id] = group.iloc[alpha - 1]['dateadded_post']
        else:
            alpha_times[topic_id] = None
    return alpha_times

# 2. Network Creation: Build Directed Graph for Each Topic
def construct_network_for_topic(group, alpha_time):
    """
    Constructs a directed graph for a single topic until alpha_time.
    """
    G = nx.DiGraph()

    # Filter posts up to alpha_time
    group = group[group['dateadded_post'] <= alpha_time]

    # Sort by timestamp and add edges
    group = group.sort_values(by='dateadded_post')
    users = group['user_id'].to_numpy()

    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            if group.iloc[j]['dateadded_post'] >= group.iloc[i]['dateadded_post']:
                G.add_edge(users[i], users[j])
    return G

# 3. Frontier Definitions for Each Cascade
def calculate_frontiers_for_topic(G, adopters, timestamps, lambda_time):
    """
    Calculate frontiers, λ frontiers, and non-adopters for a single topic.
    """
    adopters = {adopter for adopter in adopters if adopter in G}

    if not adopters:
        return set(), set(), set()

    frontiers = {neighbor for adopter in adopters for neighbor in G.successors(adopter) if neighbor not in adopters}

    lambda_frontiers = set()
    non_adopters = set()

    valid_timestamps = [timestamps[node] for node in adopters if node in timestamps]
    if not valid_timestamps:
        return frontiers, set(), set()

    origin_time = min(valid_timestamps)

    for frontier in frontiers:
        if frontier in timestamps:
            first_exposure_time = min(
                (timestamps[frontier] - timestamps[adopter]).total_seconds() / 3600
                for adopter in adopters
                if adopter in timestamps and frontier in G.successors(adopter)
            )
            if first_exposure_time <= lambda_time:
                lambda_frontiers.add(frontier)
            else:
                non_adopters.add(frontier)

    return frontiers, lambda_frontiers, non_adopters

# 4. Process All Topics
def process_all_topics(df, alpha, lambda_time):
    """
    Process all topics to calculate frontiers, λ frontiers, and non-adopters for each cascade.
    """
    alpha_times = calculate_alpha_times(df, alpha)
    roots = get_roots(df)  # Identify roots for all topics
    results = []

    for topic_id, group in df.groupby('topic_id'):
        alpha_time = alpha_times.get(topic_id)
        if alpha_time is None:
            continue

        # Construct the network
        G = construct_network_for_topic(group, alpha_time)

        # Save and visualize network for inspection
        save_net(G, topic_id)
        visualize_network(G, topic_id)

        # Get adopters and timestamps
        adopters = set(group['user_id'][:alpha])
        timestamps = {row['user_id']: row['dateadded_post'] for _, row in group.iterrows()}

        # Calculate frontiers
        frontiers, lambda_frontiers, non_adopters = calculate_frontiers_for_topic(G, adopters, timestamps, lambda_time)

        results.append({
            "topic_id": topic_id,
            "root_user": roots.get(topic_id),
            "frontiers": len(frontiers),
            "lambda_frontiers": len(lambda_frontiers),
            "non_adopters": len(non_adopters),
            "nodes": len(G.nodes()),
            "edges": len(G.edges())
        })

    return pd.DataFrame(results)

# Full Pipeline
if __name__ == "__main__":
    # Parameters for dataset loading
    forum_id = 4
    min_post_length = 10
    min_posts_per_user = 5
    min_threads_per_user = 2
    classification_threshold = 0.5

    # Load dataset
    df = load_and_filter_data(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold)

    # Define network parameters
    alpha = 10  # Number of adopters to define cascade
    lambda_time = 24  # λ in hours

    # Process topics and create networks
    results = process_all_topics(df, alpha, lambda_time)

    # Output results
    print("Final Results:")
    print(results)
    print("\nSummary Statistics:")
    print(results.describe())
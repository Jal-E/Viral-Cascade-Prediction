import os 
import pandas as pd
import networkx as nx
from sqlalchemy import create_engine

# Directories for outputs
NETWORK_DIR = "networks"
VISUALIZATION_DIR = "visualizations"
os.makedirs(NETWORK_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Database Connection
def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    """
    engine = create_engine('postgresql://postgres:1234@localhost:5432/Oct')
    return engine.connect()

# Load and Filter Data
def load_and_filter_data(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold):
    """
    Load and filter data from the database.
    """
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
    
    # Convert timestamps and filter users
    df['dateadded_post'] = pd.to_datetime(df['dateadded_post'], utc=True)
    eligible_users = df['user_id'].value_counts()[lambda x: x >= min_posts_per_user].index
    df = df[df['user_id'].isin(eligible_users)]
    eligible_thread_users = df.groupby('user_id')['topic_id'].nunique()[lambda x: x >= min_threads_per_user].index
    df = df[df['user_id'].isin(eligible_thread_users)]
    return df

# Debugging: Print parameters
def debug_print_parameters(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold, lambda_time, alpha):
    print(f"forum_id = {forum_id}")
    print(f"min_post_length = {min_post_length}")
    print(f"min_posts_per_user = {min_posts_per_user}")
    print(f"min_threads_per_user = {min_threads_per_user}")
    print(f"classification_threshold = {classification_threshold}")
    print(f"lambda_time = {lambda_time}")
    print(f"alpha = {alpha}")
    print("-" * 50)

# Debugging: Print topic details
def debug_print_topic_details(topic_id, G, adopters, alpha_times, cut_off_times, frontiers, lambda_frontiers, non_adopters):
    print(f"Processing topic {topic_id}")
    print(f"Adopters: {adopters}")
    print(f"Alpha Times: {alpha_times}")
    print(f"Cut-off Times: {cut_off_times}")
    print(f"Frontiers: {frontiers}")
    print(f"Lambda Frontiers: {lambda_frontiers}")
    print(f"Non-Adopters: {non_adopters}")
    print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
    print("-" * 50)

# Process Topics with Global Network
def process_all_topics_with_global_network(df, alpha, lambda_time):
    """
    Process topics with all threads included in the network.
    """
    specific_topics = {7182, 7192, 7193}  # Topics to debug
    results = []

    # Step 1: Create a global graph across all threads
    global_G = nx.DiGraph()
    for topic_id, group in df.groupby('topic_id'):
        users = group.sort_values(by='dateadded_post')['user_id'].tolist()
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                if users[i] != users[j]:
                    global_G.add_edge(users[i], users[j])

    # Step 2: Process each topic individually
    for topic_id, group in df.groupby('topic_id'):
        group = group.sort_values(by='dateadded_post')
        adopters = group['user_id'].head(alpha).tolist()
        timestamps = dict(zip(df['user_id'], df['dateadded_post']))

        # Collect all alpha adopter timestamps
        alpha_times = [timestamps[adopter] for adopter in adopters if adopter in timestamps]

        # Determine the cut-off time (timestamps of all alpha adopters)
        cut_off_times = {adopter: timestamps[adopter] for adopter in adopters}

        # Filter the global graph for edges within the cut-off time for each adopter
        filtered_G = nx.DiGraph()
        for u, v in global_G.edges():
            if u in timestamps and v in timestamps:
                if timestamps[u] <= max(cut_off_times.values()) and timestamps[v] <= max(cut_off_times.values()):
                    filtered_G.add_edge(u, v)

        # Identify frontiers (neighbors of adopters in the filtered graph)
        frontiers = set()
        for adopter in adopters:
            if adopter in filtered_G:
                for neighbor in filtered_G.successors(adopter):
                    # Exclude adopters themselves
                    if neighbor not in adopters:
                        frontiers.add(neighbor)

        # Classify frontiers into lambda_frontiers and non_adopters
        lambda_time_delta = pd.Timedelta(hours=lambda_time)
        lambda_frontiers = set()
        non_adopters = set()

        for node in frontiers:
            if node in timestamps:
                adopted_within_lambda = False
                node_posts_in_topic = df[(df['user_id'] == node) & (df['topic_id'] == topic_id)]
                if not node_posts_in_topic.empty:
                    node_topic_time = node_posts_in_topic['dateadded_post'].iloc[0]
                    for adopter in adopters:
                        adopter_time = timestamps[adopter]
                        # Check if the node's post time is within lambda_time after the adopter's post
                        if adopter_time <= node_topic_time <= adopter_time + lambda_time_delta:
                            adopted_within_lambda = True
                            break
                if adopted_within_lambda:
                    lambda_frontiers.add(node)
                else:
                    non_adopters.add(node)
            else:
                non_adopters.add(node)

        # Debugging outputs for specific topics
        if topic_id in specific_topics:
            debug_print_topic_details(
                topic_id=topic_id,
                G=filtered_G,
                adopters=adopters,
                alpha_times=alpha_times,
                cut_off_times=cut_off_times,
                frontiers=frontiers,
                lambda_frontiers=lambda_frontiers,
                non_adopters=non_adopters,
            )

        # Collect the results for this topic
        results.append({
            "topic_id": topic_id,
            "frontiers": len(frontiers),
            "lambda_frontiers": len(lambda_frontiers),
            "non_adopters": len(non_adopters),
            "nodes": len(filtered_G.nodes()),
            "edges": len(filtered_G.edges()),
        })

    return pd.DataFrame(results)

# Main Pipeline
if __name__ == "__main__":
    forum_id = 4
    min_post_length = 10
    min_posts_per_user = 5
    min_threads_per_user = 2
    classification_threshold = 0.5
    lambda_time = 24  # Set lambda_time to 50 hours
    alpha = 10         # Set alpha to 2, as in the sample code

    debug_print_parameters(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold, lambda_time, alpha)

    df = load_and_filter_data(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold)
    results = process_all_topics_with_global_network(df, alpha, lambda_time)
    print("Final Results:")
    print(results)
    results.to_csv("thread_network_analysis_results_debugged.csv", index=False)
    print("Results saved to `thread_network_analysis_results_debugged.csv`")

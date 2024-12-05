import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import timedelta
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

# Sort Posts by Date
def sort_posts_by_date(df):
    """
    Sort the posts within each topic by `dateadded_post` in ascending order.
    """
    return df.sort_values(by='dateadded_post')

# Validate Frontier Classification with Debugging
def validate_frontiers(df, adopters, lambda_time, topic_id, expected_frontiers):
    """
    Validate if the fresh lambda frontiers are correctly classified.
    Adds detailed debugging outputs for adopter_time and node_time comparisons.
    """
    fresh_lambda_frontiers = set()
    old_lambda_frontiers = set()

    print(f"\n--- Validating Frontiers for Topic {topic_id} ---")
    print(f"Adopters and their post times:")
    for adopter in adopters:
        adopter_time = df.loc[df['user_id'] == adopter, 'dateadded_post'].min()
        print(f"  Adopter {adopter}: {adopter_time}")

    for node in expected_frontiers:
        node_posts = df[(df['user_id'] == node) & (df['topic_id'] == topic_id)]
        
        if not node_posts.empty:
            node_time = node_posts['dateadded_post'].min()
            print(f"\nChecking Node {node}:")
            print(f"  Node Time: {node_time}")
            adopter_times = []
            for adopter in adopters:
                adopter_time = df.loc[df['user_id'] == adopter, 'dateadded_post'].min()
                time_diff = node_time - adopter_time
                adopter_times.append((adopter, adopter_time, time_diff))
                print(f"    Comparing with Adopter {adopter}:")
                print(f"      Adopter Time: {adopter_time}")
                print(f"      Time Difference (hours): {time_diff.total_seconds() / 3600:.2f}")
                if timedelta(hours=0) <= time_diff <= timedelta(hours=lambda_time):
                    print(f"      --> Node {node} adopted within lambda_time after Adopter {adopter}.")
            # Determine if node adopted within lambda_time after any adopter
            adopted_within_lambda = any(timedelta(hours=0) <= diff <= timedelta(hours=lambda_time) for _, _, diff in adopter_times)
            if adopted_within_lambda:
                fresh_lambda_frontiers.add(node)
                print(f"  --> Classified as Fresh Lambda Frontier.")
            else:
                old_lambda_frontiers.add(node)
                print(f"  --> Classified as Old Lambda Frontier.")
        else:
            print(f"\nNode {node} has no posts in Topic {topic_id}. Classified as Old Lambda Frontier.")
            old_lambda_frontiers.add(node)

    return fresh_lambda_frontiers, old_lambda_frontiers

# Construct Graph
def construct_graph(df, topic_id):
    """
    Construct the graph for the specified topic and print the sample edges.
    """
    G = nx.DiGraph()
    topic_posts = df[df['topic_id'] == topic_id]
    users = topic_posts['user_id'].tolist()
    post_times = topic_posts['dateadded_post'].tolist()
    
    for i, (u, u_time) in enumerate(zip(users, post_times)):
        for j in range(i):
            v = users[j]
            v_time = post_times[j]
            if u != v:
                G.add_edge(v, u, timestamp=v_time)
    
    sample_edges = list(G.edges(data=True))[:5]
    print(f"\nSample edges in filtered graph for topic {topic_id}: {sample_edges}")
    return G

# Main Function with Enhanced Debugging
def rerun_script_with_adjustments(df, topic_id, adopters, lambda_time, expected_frontiers):
    """
    Apply adjustments and rerun the script with enhanced debugging.
    """
    # Step 2: Sort posts
    df_sorted = sort_posts_by_date(df)
    
    # Step 3: Validate frontiers with detailed debugging
    fresh_frontiers, old_frontiers = validate_frontiers(df_sorted, adopters, lambda_time, topic_id, expected_frontiers)
    
    print(f"\nFresh lambda frontiers for topic {topic_id}: {fresh_frontiers}")
    print(f"Old lambda frontiers for topic {topic_id}: {old_frontiers}")
    
    # Step 4: Construct the graph
    G = construct_graph(df_sorted, topic_id)
    
    # Visualize the graph (optional)
    plt.figure(figsize=(12, 12))
    nx.draw_networkx(G, with_labels=False, node_size=10)
    plt.title(f"Topic {topic_id} Network Visualization")
    plt.show()

# Process Topics with Global Network (Optional: can keep or remove)
def process_all_topics_with_global_network(df, alpha, lambda_time):
    """
    Process topics with all threads included in the network.
    """
    specific_topics = {7182, 7192, 7193}  # Topics to debug
    results = []

    # Define a timezone-aware maximum timestamp
    MAX_TIMESTAMP = pd.Timestamp('9999-12-31 23:59:59', tz='UTC')

    # Step 1: Create a global graph across all threads with edge timestamps
    global_G = nx.DiGraph()
    
    # Sort the DataFrame by dateadded_post to ensure chronological order
    df_sorted = df.sort_values(by='dateadded_post')

    for topic_id, group in df_sorted.groupby('topic_id'):
        sorted_group = group.sort_values('dateadded_post')
        users = sorted_group['user_id'].tolist()
        post_times = sorted_group['dateadded_post'].tolist()
        
        # Iterate over posts in chronological order
        for i, (u, u_time) in enumerate(zip(users, post_times)):
            for j in range(i):
                v = users[j]
                v_time = post_times[j]
                if u != v:
                    # Assign the timestamp of the source user (v) as the interaction time
                    if global_G.has_edge(v, u):
                        # If edge already exists, keep the earliest timestamp
                        existing_time = global_G[v][u]['timestamp']
                        if v_time < existing_time:
                            global_G[v][u]['timestamp'] = v_time
                    else:
                        global_G.add_edge(v, u, timestamp=v_time)

    # Build a global timestamp mapping: earliest post time per user
    global_timestamps = df.groupby('user_id')['dateadded_post'].min().to_dict()

    # Step 2: Process each topic individually
    for topic_id, group in df.groupby('topic_id'):
        group = group.sort_values(by='dateadded_post')
        adopters = group['user_id'].head(alpha).tolist()
        adopter_post_times = group['dateadded_post'].head(alpha).tolist()
        adopter_time_dict = dict(zip(adopters, adopter_post_times))

        # Determine the cut-off time (timestamp of the alpha-th adopter)
        if len(adopters) == 0:
            cut_off_time = MAX_TIMESTAMP
        else:
            last_adopter = adopters[-1]
            cut_off_time = global_timestamps.get(last_adopter, MAX_TIMESTAMP)

        # Filter the global graph: include only users with earliest post <= cut_off_time
        eligible_users = {user for user, time in global_timestamps.items() if time <= cut_off_time}
        # Create the filtered graph
        filtered_G = global_G.subgraph(eligible_users).copy()

        # Identify frontiers: successors of adopters in the filtered graph, excluding adopters themselves
        frontiers = set()
        for adopter in adopters:
            if adopter in filtered_G:
                for neighbor in filtered_G.successors(adopter):
                    if neighbor not in adopters:
                        frontiers.add(neighbor)

        # Classify frontiers into lambda_frontiers and non_adopters
        lambda_time_delta = pd.Timedelta(hours=lambda_time)
        fresh_lambda_frontiers = set()
        old_lambda_frontiers = set()

        for node in frontiers:
            node_posts_in_topic = df[(df['user_id'] == node) & (df['topic_id'] == topic_id)]
            if not node_posts_in_topic.empty:
                node_topic_time = node_posts_in_topic['dateadded_post'].min()
                # Check if node adopted within lambda_time after any adopter
                adopted_within_lambda = False
                for adopter in adopters:
                    adopter_time = global_timestamps.get(adopter, MAX_TIMESTAMP)
                    if adopter_time <= node_topic_time <= adopter_time + pd.Timedelta(hours=lambda_time):
                        adopted_within_lambda = True
                        break
                if adopted_within_lambda:
                    fresh_lambda_frontiers.add(node)
                else:
                    old_lambda_frontiers.add(node)
            else:
                old_lambda_frontiers.add(node)

        # Debugging outputs for specific topics
        if topic_id in specific_topics:
            alpha_times_list = [global_timestamps.get(user, MAX_TIMESTAMP) for user in adopters]
            debug_print_topic_details(
                topic_id=topic_id,
                G=filtered_G,
                adopters=adopters,
                alpha_times=alpha_times_list,
                fresh_lambda_frontiers=fresh_lambda_frontiers,
                old_lambda_frontiers=old_lambda_frontiers,
            )

        # Collect the results for this topic
        results.append({
            "topic_id": topic_id,
            "frontiers": len(frontiers),
            "fresh_lambda_frontiers": len(fresh_lambda_frontiers),
            "old_lambda_frontiers": len(old_lambda_frontiers),
            "nodes": len(filtered_G.nodes()),
            "edges": len(filtered_G.edges()),
        })

    return pd.DataFrame(results)

# Debugging: Print parameters
def debug_print_parameters(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold, lambda_time, alpha):
    print(f"forum_id = {forum_id}")
    print(f"min_post_length = {min_post_length}")
    print(f"min_posts_per_user = {min_posts_per_user}")
    print(f"min_threads_per_user = {min_threads_per_user}")
    print(f"classification_threshold = {classification_threshold}")
    print(f"lambda_time = {lambda_time} hours")
    print(f"alpha = {alpha}")
    print("-" * 50)

# Debugging: Print topic details
def debug_print_topic_details(topic_id, G, adopters, alpha_times, fresh_lambda_frontiers, old_lambda_frontiers):
    print(f"\n--- Topic {topic_id} Details ---")
    print(f"G: DiGraph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    print(f"Alpha Adopters: {adopters}")
    print(f"Alpha Times: {alpha_times}")
    print(f"Fresh Lambda Frontiers: {fresh_lambda_frontiers}")
    print(f"Old Lambda Frontiers: {old_lambda_frontiers}")
    print("-" * 50)

# Main Pipeline
def main():
    forum_id = 4
    min_post_length = 10
    min_posts_per_user = 5
    min_threads_per_user = 2
    classification_threshold = 0.5
    lambda_time = 24  # in hours
    alpha = 10

    debug_print_parameters(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold, lambda_time, alpha)

    # Load and filter data
    df = load_and_filter_data(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold)
    print(f"Total posts after loading: {len(df)}")
    print(f"Dropped {len(df) - len(df.drop_duplicates(['user_id', 'topic_id', 'post_id']))} duplicate posts.")
    print(f"Eligible users after min_posts_per_user ({min_posts_per_user}): {df['user_id'].nunique()}")
    print(f"Eligible users after min_threads_per_user ({min_threads_per_user}): {df['user_id'].nunique()}")
    print(f"Total users after filtering: {df['user_id'].nunique()}")
    print("-" * 50)

    # Example inputs for the rerun (for Topic 7182)
    topic_id = 7182
    adopters = [44954, 44957, 44959, 44961, 44963, 44964, 44966, 44967, 44968, 44971]  # List of alpha adopters for the topic
    expected_frontiers = {47333, 47221, 46838, 50415}  # Expected fresh lambda frontiers

    # Re-run the script with the adjustments for Topic 7182
    rerun_script_with_adjustments(df, topic_id, adopters, lambda_time, expected_frontiers)

if __name__ == "__main__":
    main()

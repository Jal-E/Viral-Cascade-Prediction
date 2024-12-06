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

    # Debug: Check if topic 7193 is present and how many posts it has after filtering
    topic_post_counts = df['topic_id'].value_counts()
    if 7193 in topic_post_counts.index:
        print(f"DEBUG: Topic 7193 post count after filtering = {topic_post_counts[7193]}")
    else:
        print("DEBUG: Topic 7193 not present after filtering.")

    return df

# Debugging: Print topic details for specified topics
def debug_print_topic_details(topic_id, adopters, alpha_times, frontiers, lambda_frontiers, non_adopters):
    print(f"Processing topic {topic_id}")
    print(f"Adopters: {adopters}")
    print(f"Alpha Times: {alpha_times}")
    print(f"Frontiers (count={len(frontiers)}): {frontiers}")
    print(f"Lambda Frontiers (count={len(lambda_frontiers)}): {lambda_frontiers}")
    print(f"Non-Adopters (count={len(non_adopters)}): {non_adopters}")
    print("-" * 50)

# Process Topics with Global Network
def process_all_topics_with_global_network(df, alpha, lambda_time):
    """
    Process topics that reached alpha posts using the global network.
    Follow the strict logic described.
    """
    lambda_time_delta = pd.Timedelta(hours=lambda_time)

    # Step 1: Create a global graph across all threads
    global_G = nx.DiGraph()
    for topic_id, group in df.groupby('topic_id'):
        group_sorted = group.sort_values(by='dateadded_post')
        users_in_topic = group_sorted['user_id'].tolist()
        times_in_topic = group_sorted['dateadded_post'].tolist()
        for i in range(len(users_in_topic)):
            for j in range(i+1, len(users_in_topic)):
                u = users_in_topic[i]
                v = users_in_topic[j]
                if u != v:
                    global_G.add_edge(u, v, formed_at=times_in_topic[j])

    # Identify cascade topics: those with at least alpha posts
    cascade_topics = [t_id for t_id, grp in df.groupby('topic_id') if len(grp) >= alpha]

    # Debug: Confirm if 7193 is a cascade topic
    if 7193 in cascade_topics:
        print("DEBUG: 7193 is in cascade topics")
    else:
        print("DEBUG: 7193 not found in cascade topics, meaning it doesn't have >= alpha posts after filtering")

    results = []
    specific_topics = {7182, 7192, 7193}  # Topics to debug

    for topic_id in cascade_topics:
        group = df[df['topic_id'] == topic_id].sort_values(by='dateadded_post')
        if len(group) < alpha:
            # Not actually a cascade topic after this check
            continue

        # First alpha posters (adopters)
        adopters = group['user_id'].iloc[:alpha].tolist()
        alpha_times = group['dateadded_post'].iloc[:alpha].tolist()
        adopter_to_time = dict(zip(adopters, alpha_times))

        frontier_exposures = {}  # user -> earliest_exposure_time

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

        if topic_id in specific_topics:
            debug_print_topic_details(topic_id, adopters, alpha_times, frontiers, lambda_frontiers, non_adopters)

        results.append({
            "topic_id": topic_id,
            "frontiers": len(frontiers),
            "lambda_frontiers": len(lambda_frontiers),
            "non-adopters": len(non_adopters),
            "adopters_count": len(adopters)
        })

    return pd.DataFrame(results)

# Main Pipeline
if __name__ == "__main__":
    forum_id = 4
    min_post_length = 10
    min_posts_per_user = 5
    min_threads_per_user = 2
    classification_threshold = 0.5
    lambda_time = 24  # Lambda time in hours
    alpha = 10  # Number of adopters to consider

    df = load_and_filter_data(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold)
    results = process_all_topics_with_global_network(df, alpha, lambda_time)
    print("Final Results:")
    print(results)
    results.to_csv("thread_network_analysis_results_debugged.csv", index=False)
    print("Results saved to `thread_network_analysis_results_debugged.csv`")

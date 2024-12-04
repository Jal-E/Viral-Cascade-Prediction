import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle as pk
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

# Create Thread-Specific Network
def create_thread_network(group):
    """
    Create a directed graph (DiGraph) for a single thread.
    """
    G = nx.DiGraph()
    users = group.sort_values(by='dateadded_post')['user_id'].to_numpy()
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            if users[i] != users[j]:
                G.add_edge(users[i], users[j])
    return G

# Save Network
def save_net(G, topic_id):
    filename = os.path.join(NETWORK_DIR, f"network_topic_{topic_id}.pkl")
    with open(filename, 'wb') as f:
        pk.dump(G, f)
    print(f"Network saved to {filename}")

# Visualize Network
def visualize_network(G, topic_id):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    plt.title(f"Network Visualization for Topic {topic_id}")
    filename = os.path.join(VISUALIZATION_DIR, f"network_topic_{topic_id}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Network visualization saved to {filename}")

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
def debug_print_topic_details(topic_id, G, adopters, alpha_times, lambda_frontiers, non_adopters):
    print(f"topic_id = {topic_id}")
    print(f"G: DiGraph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    print(f"alpha_adopters = {list(adopters)}")
    print(f"alpha_times = {alpha_times}")
    print(f"lambda_frontiers = {lambda_frontiers}")
    print(f"non_adopters = {non_adopters}")
    print("-" * 50)

# Process Topics
def process_all_topics_with_debugging(df, alpha, lambda_time):
    """
    Process all topics
    """
    specific_topics = {7182, 7192, 7193}  # Topics to debug
    results = []  

    for topic_id, group in df.groupby('topic_id'):  # Group data by topic_id
        
        # Step 1: Create the directed graph for the thread
        G = create_thread_network(group)  # Build a directed graph from the posts
        save_net(G, topic_id)  # Save the network as a file
        visualize_network(G, topic_id)  # Generate and save a visualization of the network

        # Step 2: Identify adopters
        adopters = set(group.sort_values(by='dateadded_post')['user_id'][:alpha])  
        # the first 'alpha' users who posted chronologically in this thread

        # a dictionary of [user_id: post timestamp] for comparisons later
        timestamps = {row['user_id']: row['dateadded_post'] for _, row in group.iterrows()}

        # Step 3: Find the neighbors (or frontiers) of all adopters
        frontiers = {
            neighbor
            for adopter in adopters  
            if adopter in G  # exists in the graph
            for neighbor in G.successors(adopter)  # Get all (successors) of the adopter exposed to the topic but not yet adopters themselves
        }

        # Step 4: Define the time window for influence
        lambda_time_delta = pd.Timedelta(hours=lambda_time)  # Convert lambda_time (in hours) to timedelta

        # Initialize sets to classify frontiers
        lambda_frontiers = set()  # Users influenced within the allowed time window
        non_adopters = set()  # Users who are not influenced within lamda

        # Step 5: Classify each frontier based on influence timing
        for node in frontiers:
            if node in timestamps:  # If the user has posted
                # Check if the node's adoption time falls within the time window of any adopter
                adopted_within_lambda = any(
                    timestamps[adopter] <= timestamps[node] <= timestamps[adopter] + lambda_time_delta
                    for adopter in adopters  # Compare the adopter's timestamp with the node's timestamp
                )
                if adopted_within_lambda: #true?
                    lambda_frontiers.add(node)  # Add to the "lambda_frontiers" set
                else:
                    non_adopters.add(node)  # Add to the "non_adopters" set
            else:
                non_adopters.add(node)  # Add to "non_adopters" if no timestamp is available (edge case)

        # Step 6: Collect timestamps of adopters
        alpha_times_list = [
            timestamps[user] for user in adopters if user in timestamps
        ]

        # Step 7: Debugging outputs for specific topics
        if topic_id in specific_topics:  # {7182, 7192, 7193}
            debug_print_topic_details(
                topic_id=topic_id,
                G=G,
                adopters=adopters,
                alpha_times=alpha_times_list,
                lambda_frontiers=lambda_frontiers,
                non_adopters=non_adopters,
            )

        # Step 8: Store the results for the current topic
        results.append({
            "topic_id": topic_id,  
            "frontiers": len(frontiers),  
            "lambda_frontiers": len(lambda_frontiers),  
            "non_adopters": len(non_adopters), 
            "nodes": len(G.nodes()),  # Total number of nodes in the graph
            "edges": len(G.edges()),  # Total number of edges in the graph
        })

    return pd.DataFrame(results)  # to DataFrame 

# Main Pipeline
if __name__ == "__main__":
    forum_id = 4
    min_post_length = 10
    min_posts_per_user = 5
    min_threads_per_user = 2
    classification_threshold = 0.5
    lambda_time = 24
    alpha = 10

    debug_print_parameters(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold, lambda_time, alpha)

    df = load_and_filter_data(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold)
    results = process_all_topics_with_debugging(df, alpha, lambda_time)
    print("Final Results:")
    print(results)
    results.to_csv("thread_network_analysis_results_debugged.csv", index=False)
    print("Results saved to `thread_network_analysis_results_debugged.csv`")

import pandas as pd
from datetime import datetime

# Sample data representing posts and topics
data = {
    "user_id": [44961, 44963, 44964, 44966, 44967, 44968, 45063, 44959, 45030, 44976, 45185, 45186, 44995],
    "topic_id": [7182, 7182, 7182, 7182, 7182, 7182, 7182, 7182, 7192, 7192, 7193, 7193, 7193],
    "post_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "dateadded_post": [
        "2021-02-09 17:00:00", "2021-02-11 15:57:38", "2021-02-13 14:08:57",
        "2021-02-18 06:29:50", "2021-02-19 09:09:17", "2021-02-19 09:27:09",
        "2021-02-20 12:00:00", "2021-02-21 12:00:00", "2022-04-06 11:15:19",
        "2022-12-07 02:38:24", "2021-09-03 08:18:39", "2021-09-05 03:32:20", "2021-10-03 03:21:12"
    ],
    "content_post": ["text"] * 13,
    "classification2_topic": [0.8] * 13
}

# Convert to DataFrame and set proper dtypes
df = pd.DataFrame(data)
df["dateadded_post"] = pd.to_datetime(df["dateadded_post"])

import networkx as nx
from datetime import timedelta

# Function to create a directed graph for a single topic
def create_thread_network(group):
    G = nx.DiGraph()
    users = group.sort_values(by='dateadded_post')['user_id'].to_numpy()
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            if users[i] != users[j]:
                G.add_edge(users[i], users[j])
    return G

# Function to process topics with debugging
def process_topics_with_debugging(df, alpha, lambda_time, specific_topics):
    results = []

    for topic_id, group in df.groupby('topic_id'):
        if topic_id not in specific_topics:
            continue
        
        print(f"Processing topic_id: {topic_id}")
        
        # Create a directed graph
        G = create_thread_network(group)

        # Identify alpha adopters and their timestamps
        adopters = set(group.sort_values(by='dateadded_post')['user_id'][:alpha])
        timestamps = {row['user_id']: row['dateadded_post'] for _, row in group.iterrows()}

        # Calculate frontiers
        frontiers = {neighbor for adopter in adopters if adopter in G for neighbor in G.successors(adopter)}

        lambda_time_delta = timedelta(hours=lambda_time)
        lambda_frontiers = set()
        non_adopters = set()

        for node in frontiers:
            if node in timestamps:
                adopted_within_lambda = any(
                    timestamps[adopter] <= timestamps[node] <= timestamps[adopter] + lambda_time_delta
                    for adopter in adopters
                )
                if adopted_within_lambda:
                    lambda_frontiers.add(node)
                else:
                    non_adopters.add(node)
            else:
                non_adopters.add(node)

        # Debugging output
        print(f"G: DiGraph with {len(G.nodes)} nodes and {len(G.edges)} edges")
        print(f"alpha_adopters = {list(adopters)}")
        print(f"lambda_frontiers = {lambda_frontiers}")
        print(f"non_adopters = {non_adopters}")
        print("-" * 50)

        results.append({
            "topic_id": topic_id,
            "frontiers": len(frontiers),
            "lambda_frontiers": len(lambda_frontiers),
            "non_adopters": len(non_adopters),
            "nodes": len(G.nodes),
            "edges": len(G.edges)
        })

    return pd.DataFrame(results)

# Parameters
alpha = 3
lambda_time = 24
specific_topics = {7182, 7192, 7193}

# Process the topics and display results
results = process_topics_with_debugging(df, alpha, lambda_time, specific_topics)
print(results)


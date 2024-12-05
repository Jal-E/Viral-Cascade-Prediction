import pandas as pd
import networkx as nx
from datetime import datetime, timedelta

# Sample dataset
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

df = pd.DataFrame(data)
df["dateadded_post"] = pd.to_datetime(df["dateadded_post"])  # Ensure datetime format

# Parameters
alpha = 2  # Number of alpha adopters
lambda_time = 50  # Lambda time in hours 

# Step 1: Create a global network
global_G = nx.DiGraph()
for topic_id, group in df.groupby("topic_id"):
    users = group.sort_values(by="dateadded_post")["user_id"].tolist()
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            if users[i] != users[j]:
                global_G.add_edge(users[i], users[j])

# Step 2: Process each topic
results = []
for topic_id, group in df.groupby("topic_id"):
    print(f"Processing topic {topic_id}")
    
    # Extract alpha adopters and their timestamps
    group = group.sort_values(by="dateadded_post")
    adopters = group["user_id"].head(alpha).tolist()
    timestamps = dict(zip(df["user_id"], df["dateadded_post"]))  # Use df to get timestamps for all users

    # Determine the cut-off time (timestamp of the alpha-th adopter)
    if adopters:
        cut_off_time = timestamps[adopters[-1]]  # Time of the last adopter among the alpha adopters
    else:
        cut_off_time = None

    # Filter the global graph for edges within the cut-off time
    filtered_G = nx.DiGraph()
    for u, v in global_G.edges():
        # Include the edge if both nodes' timestamps are <= cut-off time
        if u in timestamps and v in timestamps:
            if timestamps[u] <= cut_off_time and timestamps[v] <= cut_off_time:
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
    lambda_time_delta = timedelta(hours=lambda_time)
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
                    if adopter_time <= node_topic_time <= adopter_time + lambda_time_delta:
                        adopted_within_lambda = True
                        break
            if adopted_within_lambda:
                lambda_frontiers.add(node)
            else:
                non_adopters.add(node)
        else:
            non_adopters.add(node)
    
    # Debugging outputs
    print(f"Adopters: {adopters}")
    print(f"Alpha Times: {[timestamps[user] for user in adopters]}")
    print(f"Cut-off Time: {cut_off_time}")
    print(f"Frontiers: {frontiers}")
    print(f"Lambda Frontiers: {lambda_frontiers}")
    print(f"Non-Adopters: {non_adopters}")
    print(f"Nodes: {len(filtered_G.nodes())}, Edges: {len(filtered_G.edges())}")
    print("-" * 50)
    
    # Store results
    results.append({
        "topic_id": topic_id,
        "frontiers": len(frontiers),
        "lambda_frontiers": len(lambda_frontiers),
        "non_adopters": len(non_adopters),
        "nodes": len(filtered_G.nodes()),
        "edges": len(filtered_G.edges())
    })

# Final Results
results_df = pd.DataFrame(results)
print("Final Results:")
print(results_df)

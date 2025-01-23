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

# Validate Lambda Frontiers
def validate_lambda_frontiers(df, validation_data, adopter_timestamps, lambda_time):
    """
    Validate the lambda_frontiers for given topics by checking if their timestamps
    fall within the lambda_time of any adopter's timestamp.
    """
    lambda_time_delta = pd.Timedelta(hours=lambda_time)
    for topic_id, data in validation_data.items():
        print(f"\nValidating topic {topic_id}")
        lambda_frontiers = data["lambda_frontiers"]
        adopters = adopter_timestamps.get(topic_id, {})
        
        for frontier in lambda_frontiers:
            frontier_posts = df[(df["user_id"] == frontier) & (df["topic_id"] == topic_id)]
            if not frontier_posts.empty:
                frontier_time = frontier_posts["dateadded_post"].iloc[0]
                valid_exposure = False
                for adopter, adopter_time in adopters.items():
                    if adopter_time <= frontier_time <= adopter_time + lambda_time_delta:
                        print(f"Frontier {frontier} posted at {frontier_time}, within lambda_time of adopter {adopter} who posted at {adopter_time}")
                        valid_exposure = True
                        break
                if not valid_exposure:
                    print(f"ERROR: Frontier {frontier} posted at {frontier_time} but did not fall within lambda_time of any adopter.")
            else:
                print(f"ERROR: No posts found for lambda_frontier {frontier} in topic {topic_id}.")

    print("Validation complete.")

# Main Pipeline
if __name__ == "__main__":
    # Parameters for data loading and lambda time
    forum_id = 4
    min_post_length = 10
    min_posts_per_user = 5
    min_threads_per_user = 2
    classification_threshold = 0.5
    lambda_time = 24  # in hours

    # Load and filter the dataset
    df = load_and_filter_data(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold)

    # Validation data
    validation_data = {
        7182: {"lambda_frontiers": {44961, 44963, 44964, 44966, 44967, 44968, 44971, 44957, 44959}},
        7192: {"lambda_frontiers": {45094, 45095, 45030, 45097, 45099, 45004, 45038, 44976, 45016}},
        7193: {"lambda_frontiers": {45186, 44995, 45189, 45191, 45192, 45193, 45180, 45055}},
    }

    # Adopters' timestamps for validation
    adopter_timestamps = {
        7182: {
            44954: pd.Timestamp('2021-01-24 08:16:40+0000', tz='UTC'),
            44957: pd.Timestamp('2021-02-02 16:32:22+0000', tz='UTC'),
            44959: pd.Timestamp('2021-02-09 08:51:23+0000', tz='UTC'),
            44961: pd.Timestamp('2021-02-09 17:00:00+0000', tz='UTC'),
            44963: pd.Timestamp('2021-02-11 15:57:38+0000', tz='UTC'),
            44964: pd.Timestamp('2021-02-13 14:08:57+0000', tz='UTC'),
            44966: pd.Timestamp('2021-02-18 06:29:50+0000', tz='UTC'),
            44967: pd.Timestamp('2021-02-19 09:09:17+0000', tz='UTC'),
            44968: pd.Timestamp('2021-02-19 09:27:09+0000', tz='UTC'),
            44971: pd.Timestamp('2021-03-22 04:29:29+0000', tz='UTC'),
        },
        7192: {
            45093: pd.Timestamp('2022-04-06 11:15:19+0000', tz='UTC'),
            45094: pd.Timestamp('2022-04-06 14:22:06+0000', tz='UTC'),
            45095: pd.Timestamp('2022-12-07 00:36:55+0000', tz='UTC'),
            44976: pd.Timestamp('2022-12-07 02:38:24+0000', tz='UTC'),
            45004: pd.Timestamp('2022-12-08 10:09:29+0000', tz='UTC'),
        },
        7193: {
            45185: pd.Timestamp('2021-09-03 08:18:39+0000', tz='UTC'),
            45186: pd.Timestamp('2021-09-05 03:32:20+0000', tz='UTC'),
            45055: pd.Timestamp('2021-09-24 20:13:59+0000', tz='UTC'),
            44995: pd.Timestamp('2021-10-03 03:21:12+0000', tz='UTC'),
        },
    }

    # Validate lambda frontiers
    validate_lambda_frontiers(df, validation_data, adopter_timestamps, lambda_time)

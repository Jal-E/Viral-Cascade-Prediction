import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# Database connection
def get_db_connection():
    engine = create_engine('postgresql://postgres:1234@localhost:5432/Oct')
    return engine.connect()

# Load and filter data
def load_and_filter_data(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold):
    conn = get_db_connection()
    query = """
    SELECT posts.user_id, posts.topic_id, posts.post_id, posts.dateadded_post, LENGTH(posts.content_post) AS post_length, 
           topics.classification2_topic
    FROM posts
    INNER JOIN topics ON posts.topic_id = topics.topic_id
    WHERE topics.forum_id = %s AND topics.classification2_topic >= %s
    """
    df = pd.read_sql(query, conn, params=(forum_id, classification_threshold))
    conn.close()

    # Apply filters
    df['dateadded_post'] = pd.to_datetime(df['dateadded_post'], utc=True)
    df = df[df['post_length'] >= min_post_length]
    user_post_counts = df['user_id'].value_counts()
    df = df[df['user_id'].isin(user_post_counts[user_post_counts >= min_posts_per_user].index)]
    user_thread_counts = df.groupby('user_id')['topic_id'].nunique()
    df = df[df['user_id'].isin(user_thread_counts[user_thread_counts >= min_threads_per_user].index)]

    return df

# Parameters for dataset loading
forum_id = 4
min_post_length = 10
min_posts_per_user = 5
min_threads_per_user = 2
classification_threshold = 0.5

# Load dataset globally for import
df = load_and_filter_data(forum_id, min_post_length, min_posts_per_user, min_threads_per_user, classification_threshold)

def explore_timestamps(df):
    """
    Explore the dateadded_post column for anomalies, missing values, and distribution.
    """
    print("\nExploring Timestamps...")

    # Check for missing values
    missing_count = df['dateadded_post'].isnull().sum()
    print(f"Missing dateadded_post values: {missing_count}")

    # Check date range
    min_date = df['dateadded_post'].min()
    max_date = df['dateadded_post'].max()
    print(f"Date range: {min_date} to {max_date}")

    # Check sorting
    sorted_correctly = df['dateadded_post'].is_monotonic_increasing
    print(f"Is the dataset sorted by dateadded_post? {'Yes' if sorted_correctly else 'No'}")

    # Plot distribution
    plt.figure(figsize=(10, 5))
    df['dateadded_post'].hist(bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of dateadded_post')
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    print("Dataset Overview:")
    print(df.head())
    print("Dataset Info:")
    print(df.info())
    print("Date Range:", df['dateadded_post'].min(), "to", df['dateadded_post'].max())
    print("Dataset loaded and available as `df`.")
    
    # Timestamp Exploration
    explore_timestamps(df)

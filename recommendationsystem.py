import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Initialize FastAPI app
app = FastAPI()



# Load Data
posts = pd.read_csv('allposts.csv')
users = pd.read_csv("alluser.csv")
likedposts = pd.read_csv("liked_posts.csv")
viewedposts = pd.read_csv("viewed_posts.csv")

#  Preprocess Interaction Data
def preprocess_interactions(likedposts, viewedposts):
    likedposts['interaction_type'] = 'like'
    viewedposts['interaction_type'] = 'view'
    return pd.concat([likedposts, viewedposts], ignore_index=True)

interaction_df = preprocess_interactions(likedposts, viewedposts)

# Group Interactions by User
def group_user_interactions(interaction_df):
    return interaction_df.groupby('user_id')['post_id'].apply(list).reset_index()

user_interactions = group_user_interactions(interaction_df)

# Extract Text from Post Summary
def extract_text(summary):
    try:
        summary_dict = ast.literal_eval(summary)
        text_elements = []

        def extract_text_recursively(data):
            if isinstance(data, dict):
                for value in data.values():
                    extract_text_recursively(value)
            elif isinstance(data, list):
                for item in data:
                    extract_text_recursively(item)
            elif isinstance(data, str):
                text_elements.append(data)

        extract_text_recursively(summary_dict)
        return " ".join(text_elements)
    except (ValueError, SyntaxError):
        return summary

posts['extracted_text'] = posts['post_summary'].apply(extract_text)

#  Combine Text Fields for TF-IDF
def combine_text_fields(posts):
    posts['combined_text'] = posts['title'].fillna('') + " " + posts['extracted_text'].fillna('')
    return posts

posts = combine_text_fields(posts)

# Process Categories and Add Category ID
def process_categories(posts):
    posts['category'] = posts['category'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    posts['category_id'] = posts['category'].apply(lambda x: x['id'] if isinstance(x, dict) and 'id' in x else None)
    return posts

posts = process_categories(posts)

#  TF-IDF Vectorization
def compute_tfidf(posts):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    return vectorizer.fit_transform(posts['combined_text'])

tfidf_matrix = compute_tfidf(posts)

#  Scale Numerical Features
def scale_numerical_features(posts):
    numerical_features = ['comment_count', 'upvote_count', 'view_count', 'rating_count', 'category_id']
    scaler = MinMaxScaler()
    return scaler.fit_transform(posts[numerical_features].fillna(0))

scaled_numerical = scale_numerical_features(posts)

#  Combine TF-IDF and Numerical Features
def combine_features(tfidf_matrix, scaled_numerical):
    return np.hstack([tfidf_matrix.toarray(), scaled_numerical])

content_matrix = combine_features(tfidf_matrix, scaled_numerical)

#  Compute Similarity Matrix
def compute_similarity_matrix(content_matrix):
    return cosine_similarity(content_matrix)

similarity_matrix = compute_similarity_matrix(content_matrix)

import numpy as np

def get_recommendations(user_id: int, category_id: int = None, mood: str = None, top_n: int = 10):
    """
    Recommends posts for a user based on a similarity matrix.
    
    Parameters:
    - user_id: ID of the user for whom recommendations are made.
    - category_id: (Optional) Filter recommendations by category.
    - mood: (Optional) Used for cold-start recommendation when the user has no history.
    - top_n: Number of recommendations to return.
    
    Returns:
    - List of recommended post IDs.
    """
    # Get the posts liked or viewed by the user
    user_interactions = likedposts[likedposts['user_id'] == user_id]['post_id'].tolist()
    
    if user_interactions:
        # Aggregate similarities from user's interacted posts
        post_indices = [posts[posts['id'] == pid].index[0] for pid in user_interactions if pid in posts['id'].tolist()]
        post_similarities = np.sum(similarity_matrix[post_indices], axis=0)
        
        # Rank posts by similarity
        recommendations = np.argsort(-post_similarities)  # Negative for descending order
        
        # Filter out already interacted posts and by category if provided
        recommendations = [posts.iloc[i]['id'] for i in recommendations 
                           if posts.iloc[i]['id'] not in user_interactions and 
                           (category_id is None or posts.iloc[i]['category_id'] == int(category_id))]
    else:
        # Cold-start recommendation: Use category or mood if available
        if category_id:
            recommendations = posts[posts['category_id'] == int(category_id)]['id'].tolist()
        elif mood:
            # Placeholder logic: Recommend based on mood keywords in the title or summary
            recommendations = posts[posts['title'].str.contains(mood, case=False, na=False) | 
                                     posts['post_summary'].str.contains(mood, case=False, na=False)]['id'].tolist()
        else:
            # Default fallback: Recommend trending posts (e.g., by view count)
            recommendations = posts.sort_values(by='view_count', ascending=False)['id'].tolist()
    
    # Return top N recommendations
    links = []  # List to store video links
    for i in recommendations[:top_n]:
      # Check if the post exists in the DataFrame and get its video link
      video_link = posts.loc[posts['id'] == i, 'video_link'].iloc[0] if not posts[posts['id'] == i].empty else None
      if video_link:
        links.append(video_link)


    return links



# API Endpoint 1: `username`, `category_id`, and `mood`
@app.get("/feed")
def feed(username: str, category_id: str = None, mood: str = None):
    user_id = users.loc[users['username'] == username, 'id'].iloc[0] if not users[users['username'] == username].empty else None
    if user_id in user_interactions['user_id'].values:
        # Existing user: personalized recommendations
        recommended_posts = get_recommendations(user_id, category_id, mood)
    else:
        # New user: handle cold start
        recommended_posts = get_recommendations(user_id,mood=mood)

    return {"recommended_posts": recommended_posts}

# API Endpoint 2: `username` and `category_id`
@app.get("/feed_with_category")
def feed_with_category(username: str, category_id: str):
    user_id = users.loc[users['username'] == username, 'id'].iloc[0] if not users[users['username'] == username].empty else None
    recommended_posts = get_recommendations(user_id, category_id)
    return {"recommended_posts": recommended_posts}

# API Endpoint 3: `username` only
@app.get("/feed_by_username")
def feed_by_username(username: str,mood=None):
    user_id = users.loc[users['username'] == username, 'id'].iloc[0] if not users[users['username'] == username].empty else None
    if user_id in user_interactions['user_id'].values:
        # Existing user: personalized recommendations
        recommended_posts = get_recommendations(user_id)
    else:
        # New user: handle cold start
        recommended_posts = get_recommendations(user_id,mood=mood)

    return {"recommended_posts": recommended_posts}
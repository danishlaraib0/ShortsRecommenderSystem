import requests
import pandas as pd

TOKEN = "flic_6e2d8d25dc29a4ddd382c2383a903cf4a688d1a117f6eb43b35a1e7fadbb84b8"
HEADERS = {"Flic-Token": TOKEN}

def fetch_viewed_posts():
    url = "https://api.socialverseapp.com/posts/view?page=1&page_size=1000&resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def fetch_liked_posts():
    url = "https://api.socialverseapp.com/posts/like?page=1&page_size=1000&resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def fetch_inspired_posts():
    url = "https://api.socialverseapp.com/posts/inspire?page=1&page_size=1000&resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def fetch_rated_posts():
    url = "https://api.socialverseapp.com/posts/rating?page=1&page_size=1000&resonance_algorithm=resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def fetch_all_posts():
    url = "https://api.socialverseapp.com/posts/summary/get?page=1&page_size=1000"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def fetch_all_users():
    url = "https://api.socialverseapp.com/users/get_all?page=1&page_size=1000"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def save_data_to_csv(data, file_name):
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)

from src.data_collection import fetch_viewed_posts, fetch_liked_posts, fetch_inspired_posts, fetch_rated_posts, fetch_all_posts, fetch_all_users, save_data_to_csv

# Fetch and save viewed posts
viewed_posts = fetch_viewed_posts()
save_data_to_csv(viewed_posts.get("posts", []), "viewed_posts.csv")

# Fetch and save liked posts
liked_posts = fetch_liked_posts()
save_data_to_csv(liked_posts.get("posts", []), "liked_posts.csv")

# Fetch and save inspired posts
inspired_posts = fetch_inspired_posts()
save_data_to_csv(inspired_posts.get("posts", []), "inspired_posts.csv")

# Fetch and save rated posts
rated_posts = fetch_rated_posts()
save_data_to_csv(rated_posts.get("posts", []), "rated.csv")

# Fetch and save all posts
all_posts = fetch_all_posts()
save_data_to_csv(all_posts.get("posts", []), "allposts.csv")

# Fetch and save all users
all_users = fetch_all_users()
save_data_to_csv(all_users.get("users", []), "alluser.csv")

# ShortsRecommenderSystem
This project implements a content-based recommender system that suggests posts to users based on their past interactions. The system uses TF-IDF and cosine similarity for post recommendations.


To run this 
Download the dataset folder and recommendationsystem.py and set the datset path of csv file then 
open terminal and use uvicorn recommendationsystem:main --reload

and on browser we can call APIs
e.g. http://127.0.0.1:8000/feed_by_username?username=afrobeezy

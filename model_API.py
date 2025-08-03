from fastapi import FastAPI
from pymongo import MongoClient
from NeuMF import Recommender_system
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model #type: ignore
import os
import pickle
trained_model = None
dataset_cache = None
model_path = r"c:\Users\ygala\OneDrive\Desktop\recommender system/neumf_model.h5"
dataset_cache_path = r"c:\Users\ygala\OneDrive\Desktop\recommender system\dataset_cache.pkl"
if os.path.exists(model_path):
    trained_model = load_model(model_path)

if os.path.exists(dataset_cache_path):
    with open(dataset_cache_path, "rb") as f:
        dataset_cache = pickle.load(f)

recommender=Recommender_system()
app=FastAPI()

Url='mongodb+srv://admin:nHLfecsVHTG3Hkgu@cluster0.4pvvsk6.mongodb.net/Recommender_system?retryWrites=true&w=majority'
client=MongoClient(Url)
movie_db=client['Recommender_system']['movies']
ratings_db=client['Recommender_system']['ratings']
ratings=pd.read_csv('ratings.csv')
movies=pd.read_csv('movies.csv')
ratings_dict=ratings.to_dict(orient='records')
movies_dict=movies.to_dict(orient='records')

def clean_mongo_docs(docs):
    return [{k: v for k, v in doc.items() if k != '_id'} for doc in docs]

@app.post('/train_model')
def train_model():
    global trained_model, dataset_cache

    if ratings_db.count_documents({}) == 0:
        ratings_db.insert_many(ratings_dict)
    if movie_db.count_documents({}) == 0:
        movie_db.insert_many(movies_dict)

    d1 = clean_mongo_docs(list(ratings_db.find()))
    d2 = clean_mongo_docs(list(movie_db.find()))

    trained_model, dataset_cache, message = recommender.Train(d1, d2)

    trained_model.save(model_path)
    with open(dataset_cache_path, "wb") as f:
        pickle.dump(dataset_cache, f)

    return {"message": message}


@app.get('/predict/{id}')
def predict(id: int):
    global trained_model, dataset_cache

    if trained_model is None or dataset_cache is None:
        return {"error": "Model not trained yet. Call /train_model first."}

    d2 = clean_mongo_docs(list(movie_db.find()))
    dataset_df = pd.DataFrame(dataset_cache)
    movies_df = pd.DataFrame(d2)

    all_movies = set(movies_df['movieId'].unique())
    seen_movies = set(dataset_df[dataset_df['userId'] == id]['movieId'])
    unseen_movies = list(all_movies - seen_movies)

    # Get predictions
    predicts = recommender.recommended_for_user(unseen_movies, id, trained_model)
    top_recommendations = predicts.head(5)

    top_movie_ids = top_recommendations['movieId'].tolist()
    predicted_ratings = top_recommendations['predicted_rating'].tolist()

    recommended = movies_df[movies_df['movieId'].isin(top_movie_ids)].copy()
    recommended['predicted_rating'] = predicted_ratings

    recommended = movies_df[movies_df['movieId'].isin(top_movie_ids)].copy()
    recommended = pd.merge(recommended, top_recommendations, on='movieId')

    return recommended[['movieId', 'title', 'predicted_rating']].sort_values(
        by='predicted_rating', ascending=False
    ).to_dict(orient='records')








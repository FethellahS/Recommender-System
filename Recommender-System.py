import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors

# Load MovieLens dataset (you can use the 'ml-100k' dataset from http://www.grouplens.org/datasets/movielens/)
movies = pd.read_csv('movies.csv')  # File should contain movieId and title
ratings = pd.read_csv('ratings.csv')  # File should contain userId, movieId, and rating

# Pivot the data to create a user-item matrix
user_movie_ratings = ratings.pivot(index='userId', columns='movieId', values='rating')

# Fill missing values with 0
user_movie_ratings = user_movie_ratings.fillna(0)

# Calculate similarity matrix using cosine similarity
movie_similarity = cosine_similarity(user_movie_ratings.T)
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)

def get_movie_recommendations(movie_id, top_n=10):
    similar_movies = movie_similarity_df[movie_id].sort_values(ascending=False)
    similar_movies = similar_movies[similar_movies.index != movie_id]  # Remove the movie itself
    return similar_movies.head(top_n)

# Example usage
movie_id = 1  # Example movieId
recommendations = get_movie_recommendations(movie_id)
print("Recommended movies for movieId", movie_id)
print(recommendations)

# To get recommendations for a user based on their ratings
def get_user_recommendations(user_id, top_n=10):
    user_ratings = user_movie_ratings.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0].index
    unrated_movies = user_movie_ratings.columns.difference(rated_movies)

    recommendations = pd.Series()
    for movie in unrated_movies:
        similar_movies = get_movie_recommendations(movie)
        weighted_scores = similar_movies[similar_movies.index.isin(rated_movies)]
        recommendations[movie] = weighted_scores.dot(user_ratings[rated_movies]) / weighted_scores.sum()

    return recommendations.sort_values(ascending=False).head(top_n)

# Example usage
user_id = 1  # Example userId
user_recommendations = get_user_recommendations(user_id)
print("Recommended movies for userId", user_id)
print(user_recommendations)

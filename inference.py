import numpy as np
import pandas as pd
import torch


def create_observation(userId, watchedMovies, ratings, rec):
    """
    Format the observation for the model.
    """
    userIds = [userId for i in range(185)]
    
    for i in range(185):
        if i not in watchedMovies:
            watchedMovies.append(i)
    
    ratings = ratings + [0.0]*(185-len(ratings))
    
    movieEmbeds = []
    for movie_index in watchedMovies:
        for key, movie in rec.movies_map.items():
            if movie.encodedMovieId == movie_index:
                movieEmbeds.append(movie.descEmbedding)     
    
    obs = {
        "user_indices": np.array(userIds), 
        "movie_indices": np.array(watchedMovies), 
        "movie_ratings": np.array(ratings), 
        "movie_embeddings": np.array(movieEmbeds)
        }
    
    return obs

def calculate_similarity(unknown_movies, unknown_ratings, known_movies, known_ratings):
    """
    Calculate similarity between two users based on common movies and ratings.    
    """
    common_movies = set(unknown_movies).intersection(known_movies)
    common_movies_list = list(common_movies)
    
    common_ratings = [unknown_ratings[unknown_movies.index(movie)] for movie in common_movies_list]
    known_common_ratings = [known_ratings[known_movies.index(movie)] for movie in common_movies_list]

    # Calculate similarity based on common movies and ratings
    similarity = np.sum(np.array(common_ratings) == np.array(known_common_ratings))
    return similarity

def find_most_similar_user(unknown_movies, unknown_ratings, known_users_data):
    """
    Find the most similar known user for the unknown user based on common movies and ratings.
    """
    similarities = []
    for user_id, user_data in known_users_data.items():
        known_movies = [movie_id for movie_id, _ in user_data]
        known_ratings = [rating for _, rating in user_data]
        similarity = calculate_similarity(unknown_movies, unknown_ratings, known_movies, known_ratings)
        similarities.append((user_id, similarity))
    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the ID of the most similar known user
    return similarities[0][0]


def recommend_movies(model, userId, watchedMovies, ratings, rec):
    """
    Recommend movies for the based on observation.
    """
    def movieid_to_index(movieid):
        return rec.movies_map[movieid].encodedMovieId

    def get_known_user_data():
        users_map={}
        for user, user_object in rec.users_map.items():
            users_map[user] = user_object.to_dict()
            
        df = pd.DataFrame(users_map).T.drop(['userId'], axis=1).rename(columns={'encodedUserId': 'userId'})
        df['movieRating'] = df['movieRating'].apply(lambda x: [(movieid_to_index(movieid), rating) for movieid, rating in x])
        df = df.set_index('userId')
        known_users_data = df.to_dict()['movieRating']
        return known_users_data
    
    known_users = get_known_user_data()
    
    if any(movie > 185 for movie in watchedMovies):
        watchedMovies = [movieid_to_index(movieid) for movieid in watchedMovies]
    
    if userId not in known_users.keys():
        userId = find_most_similar_user(watchedMovies, ratings, known_users)
        
        # most_similar_user_data = known_users[userId]
        # most_similar_user_movies = [movie_id for movie_id, _ in most_similar_user_data]
        # most_similar_user_ratings = [rating for _, rating in most_similar_user_data]

    obs = create_observation(userId, watchedMovies, ratings, rec)
    # print(obs)
    action = model.predict(obs, deterministic=True)[0]
    
    action[obs['movie_ratings'] > 0] = float('-inf')
    action = torch.tensor(action, dtype=torch.float32)
    _, top_indices = torch.topk(action, 5)
    top_movie_indices = obs['movie_indices'][top_indices]
    top_movie_ids = []
    for movie_index in top_movie_indices:
        for key, movie in rec.movies_map.items():
            if movie.encodedMovieId == movie_index:
                top_movie_ids.append(movie.movieId)
    
    return np.array(top_movie_ids)
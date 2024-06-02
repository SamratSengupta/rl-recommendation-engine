import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch

class RecommendationEnv(gym.Env):
    def __init__(self, user_data, movies_data, total_users=999, movie_embedding_dim=100):
        super(RecommendationEnv, self).__init__()
        
        self.user_data = user_data.to_dict('records')[0]
        self.movies_data = movies_data
        self.total_users = total_users
        self.movie_embedding_dim = movie_embedding_dim
        
        self.max_item_len = len(self.user_data['user_watched_movies']) + len(self.user_data['user_unwatched_movies'])
        
        self.action_space = spaces.Box(low=0, high=1, shape=(self.max_item_len,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "user_indices": spaces.Box(low=0, high=total_users, shape=(self.max_item_len,), dtype=int),
            "movie_indices": spaces.Box(low=0, high=self.max_item_len-1, shape=(self.max_item_len,), dtype=int),           
            "movie_embeddings": spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.max_item_len, self.movie_embedding_dim), dtype=float),
            "movie_ratings": spaces.Box(low=0.0, high=5.0, shape=(self.max_item_len,), dtype=float)
        })

        self.feedback = {"liked_movies": [], "disliked_movies": [], "new_select_movies": []}

    def set_user_data(self, user_data):
        self.user_data = user_data

    def set_current_feedback(self, feedback_movies):
        self.feedback = feedback_movies

    def calculate_rewards(self):
        reward = 0
        for movie in self.feedback.get('liked_movies', []):
            reward += 5
        for movie in self.feedback.get('disliked_movies', []):
            reward -= 5
        for movie in self.feedback.get('new_select_movies', []):
            reward += 3
        return reward

    def step(self, action):
        print(f"Action taken: {action}")
        
        state = self.get_current_state()
        self.set_current_feedback(self.get_user_feedback(action))
        
        reward = self.calculate_rewards()
        self.update_user_data()
        
        done = True
        
        return state, reward, done, {}

    def reset(self):
        return self.get_current_state()

    def get_current_state(self):
        user_index = int(self.user_data['user_index'])
        movie_indices = np.arange(self.max_item_len)
        movie_embeddings = self.movies_data.loc[movie_indices, 'movie_embedding'].values
        movie_embeddings = np.array([emb.tolist() for emb in movie_embeddings])
        movie_embeddings.reshape(185, 100)
        
        movie_ratings = np.zeros(self.max_item_len)
        
        watched_movies = self.user_data['user_watched_movies']
        ratings = self.user_data['user_watched_ratings']
        
        for i, movie in enumerate(watched_movies):
            movie_index = self.movies_data.index[self.movies_data['movie_index'] == movie].tolist()[0]
            movie_ratings[movie_index] = ratings[i]

        state = {
            "user_indices": np.full(self.max_item_len, user_index),
            "movie_indices": np.array(movie_indices),
            "movie_embeddings": np.array(movie_embeddings),
            "movie_ratings": np.array(movie_ratings)
        }

        # Check for NaN values in the state
        for key, value in state.items():
            if np.isnan(value).any():
                print(f"NaN detected in state {key}: {value}")
                raise ValueError(f"NaN detected in state {key}")

        return state

    def get_user_feedback(self, action):
        feedback = self.feedback
        return feedback
    
    def update_user_data(self):
        feedback = self.feedback
        watched_movies = self.user_data['user_watched_movies']
        watched_movies.extend(feedback['liked_movies'])
        watched_movies.extend(feedback['disliked_movies'])
        watched_movies.extend(feedback['new_select_movies'])
        
        ratings = self.user_data['user_watched_ratings']
        
        for movie in watched_movies:
            if movie in feedback['liked_movies']:
                ratings.append(5.0)
            elif movie in feedback['disliked_movies']:
                ratings.append(-5.0)
            elif movie in feedback['new_select_movies']:
                ratings.append(3.0)

        self.user_data['user_watched_movies'] = watched_movies
        unwatched_movies = [m for m in self.user_data['user_unwatched_movies'] if m not in watched_movies]
        self.user_data['user_unwatched_movies'] = unwatched_movies
        self.user_data['user_watched_ratings'] = ratings
        
        print(f"Updated user data: {self.user_data}")

def update_user_data(self):
    feedback = self.feedback
    watched_movies = self.user_data['user_watched_movies']
    watched_movies.extend(feedback['liked_movies'])
    watched_movies.extend(feedback['disliked_movies'])
    watched_movies.extend(feedback['new_select_movies'])

    ratings = self.user_data['user_watched_ratings']

    for movie in watched_movies:
        if movie in feedback['liked_movies']:
            ratings.append(5.0)
        elif movie in feedback['disliked_movies']:
            ratings.append(-5.0)
        elif movie in feedback['new_select_movies']:
            ratings.append(3.0)

    self.user_data['user_watched_movies'] = watched_movies
    unwatched_movies = [m for m in self.user_data['user_unwatched_movies'] if m not in watched_movies]
    self.user_data['user_unwatched_movies'] = unwatched_movies
    self.user_data['user_watched_ratings'] = ratings

    print(f"Updated user data: {self.user_data}")



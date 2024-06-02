import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from online_env import RecommendationEnv
from online_trainer import Online_Recommendation_Trainer
from IPython.display import display

from stable_baselines3.common.logger import configure
new_logger = configure("logs", ["stdout", "csv", "tensorboard"])

import warnings
warnings.filterwarnings("ignore")

class RecommendationService:
    def __init__(self, user_id):
        
        self.user_id = int(user_id)  
        
        self.agent = PPO.load('./saved/ppo_recommender_10e.zip')    # Load the trained agent
        ### reset agent params wrt new env ###        
        self.agent.n_steps=1
        self.agent.n_epochs=1

        self.agent.set_logger(new_logger)
        # Load User data and preprocess it
        user_movie_data = pd.read_csv('./data/log/users_data.csv').drop(columns='Unnamed: 0')
        user_movie_data['user_watched_movies'] = user_movie_data['user_watched_movies'].apply(lambda x: list(map(int, x[1:-1].split(','))))
        user_movie_data['user_watched_ratings'] = user_movie_data['user_watched_ratings'].apply(lambda x: list(map(float, x[1:-1].split(','))))
        user_movie_data['recMovies'] = user_movie_data['recMovies'].apply(lambda x: list(map(int, x[1:-1].split(','))))
        user_movie_data['gtMovies'] = user_movie_data['gtMovies'].apply(lambda x: list(map(int, x[1:-1].split(','))))
        user_movie_data = user_movie_data.drop(columns=['recMovies', 'gtMovies'])
        user_movie_data['user_unwatched_movies'] = user_movie_data.apply(lambda x: [m for m in range(0, 185) if m not in x['user_watched_movies']], axis=1)
        # Get user data for given user id
        user_data = user_movie_data[user_movie_data['user_id'] == self.user_id]

        # Load Movies data and preprocess it
        movies_data = pd.read_csv('./data/log/movies_data.csv').drop(columns='Unnamed: 0')
        movies_data['movie_embedding'] = movies_data['movie_embedding'].apply(lambda x: np.array(x[1:-1].replace('\n', '').split(), dtype=np.float64))

        # Create the environment and trainer
        self.env = RecommendationEnv(user_data, movies_data)
        self.reco_trainer = Online_Recommendation_Trainer(self.agent, self.env)
        
        # Initialize the recommended, watched and unwatched movies
        self.reset_recommended_movies()
        self.set_unwatched_movies()
        self.set_watched_movies()

    def reset_recommended_movies(self):
        recommended_movies = self.reco_trainer.recommend_movies()
        print('recommended movies ::: ',recommended_movies)
        self.recommended_movies = recommended_movies

        
    def set_recommended_movies(self):
        recommended_movies = self.reco_trainer.online_update(num_steps=1)
        self.recommended_movies = recommended_movies    
        
    def set_watched_movies(self):
        self.watched_movies = self.env.user_data['user_watched_movies']

    def set_unwatched_movies(self):
        self.unwatched_movies = [m for m in self.env.user_data['user_unwatched_movies'] if m not in self.get_recommended_movies()]

    def get_recommended_movies(self):
        return self.recommended_movies
    
    def get_watched_movies(self):
        return self.watched_movies
    
    def get_unwatched_movies(self):
        return self.unwatched_movies
    
    def submit_feedback(self, submit_movies):
        # watched_movies = self.get_watched_movies() + [m for m in submit_movies['recommended_movies'].keys()] + [m for m in submit_movies['selected_movies']]
        # unwatched_movies = submit_movies['user_unwatched_movies']
        # ratings = self.env.user_data['user_watched_ratings']
        
        # for movie in watched_movies:
        #     # rating = 0.0
        #     if movie in [m for m, n in submit_movies['recommended_movies'].items() if n == 1]:
        #         ratings.append(5.0)
        #     elif movie in [m for m, n in submit_movies['recommended_movies'].items() if n != 1]:
        #         ratings.append(-5.0)
        #     elif movie in [m for m in submit_movies['selected_movies']]:
        #         ratings.append(3.0)
        
        # self.env.set_user_data({
        #     'user_id': self.user_id,
        #     'user_index': self.env.user_data['user_index'],
        #     'user_watched_movies': watched_movies,
        #     'user_unwatched_movies': unwatched_movies,
        #     'user_watched_ratings': ratings
        # })
        
        feedback = {
            'liked_movies': [m for m, n in submit_movies['recommended_movies'].items() if n == 1],
            'disliked_movies': [m for m, n in submit_movies['recommended_movies'].items() if n != 1],
            'new_select_movies': [m for m in submit_movies['selected_movies']]
        }
        self.env.set_current_feedback(feedback)
        #self.reco_trainer.online_update(num_steps=1)
        
        self.set_recommended_movies()
        self.set_watched_movies()
        self.set_unwatched_movies()
    
    
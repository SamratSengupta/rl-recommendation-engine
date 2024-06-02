import torch

import pandas as pd
import numpy as np

import gym
from gym import spaces
from sklearn.metrics.pairwise import cosine_similarity
from stable_baselines3.common.env_util import make_vec_env

from state import State
 
class RecommenderEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    def __init__(self, recommender):
        super(RecommenderEnv, self).__init__()
        self.recommender = recommender
        self.current_user_index = 0
        self.current_sequence_index = 0
        
        self.movie_embedding_dim = 100
        self.render_mode = 'human'
        self.done = False
        self.num_reco = 5
        self.batch_size = len(self.recommender.state_sequences[self.current_user_index][self.current_sequence_index]['movie_indices'])       
        self.max_item_len=len(self.recommender.movies_data.movieId.unique().tolist()) 
        
        self.action_space = spaces.Box(low=0, high=1, shape=(self.max_item_len,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "user_indices": spaces.Box(low=0, high=len(recommender.users_map), shape=(self.max_item_len,), dtype=int),           
            "movie_indices": spaces.Box(low=0, high=len(recommender.movies_map), shape=(self.max_item_len,), dtype=int),           
            "movie_embeddings": spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.max_item_len, self.movie_embedding_dim), dtype=float),
            "movie_ratings": spaces.Box(low=0.0, high=5.0, shape=(self.max_item_len,), dtype=float)
        })
        
        self.rewards_log = []
    
    def __get_movie_id(self, movie_index):
        for key, movie in self.recommender.movies_map.items():
            if movie.encodedMovieId == movie_index:
                return movie.movieId
        return None

    def _get_embeddings_for_sequence(self, sequence):
        """
        Extracts the embeddings of the movies in a given sequence.
        :param sequence: The sequence of movies.
        :param recommender: The Recommender object.
        :return: Embeddings of the movies in the sequence.
        """
        embeddings = np.array([self.recommender.movies_map[movie_id].descEmbedding for movie_id in sequence])
        return embeddings
    
    
    def _compute_reward(self, predicted_embeddings, ground_truth_embeddings, k1=10, k2=2):
        
        num_gt_embeddings = len(ground_truth_embeddings)
        num_predicted_embeddings = len(predicted_embeddings)
        rewards = []
        
        for n, predicted_embedding in enumerate(predicted_embeddings):
            similarities = cosine_similarity([predicted_embedding], ground_truth_embeddings)[0]
            max_sim_index = np.argmax(similarities)
            max_similarity = similarities[max_sim_index]
            current_similarity = similarities[n]
            
            if max_similarity > 0.9:
                ratio = max_similarity / current_similarity
                reward = ratio * k1
            else: 
                reward = - k2 
                
            rewards.append(reward)

        total_reward = sum(rewards)    
        # total_reward = random.uniform(-15, 15)
        
        return total_reward
        
 
    def get_rewards(self, action, state):
        """
        Computes the reward for the given action and state based on the cosine-weighted NDCG metric.
        """ 
        action[state['movie_ratings'] > 0] = float('-inf')                      # Invalidate scores for visited movies
        action = torch.tensor(action, dtype=torch.float32)                      
        _, top_indices = torch.topk(action, 5)                                  # Get top-5 movie indices
        top_movie_indices = state['movie_indices'][top_indices]
        top_movie_ids = [self.__get_movie_id(movie_index) for movie_index in top_movie_indices] # Convert indices to movie IDs
        pred_embeddings = self._get_embeddings_for_sequence(top_movie_ids)      # Get embeddings for the top-5 predicted movies
        
        current_seq = self.recommender.sequences['sequences'][self.current_user_index][self.current_sequence_index]
        next_pos_seq = np.array(current_seq['next_pos_samples'])                # Get the next positive sequence
        gt_embeddings = self._get_embeddings_for_sequence(next_pos_seq)         # Get embeddings for the next positive sequence
        
        # print('pred_movies_ids ',top_movie_ids)
        # print('pred_embeddings shape ',pred_embeddings.shape)
        # print('gt_movie_ids ',next_pos_seq)
        # print('gt_embeddings shape ',gt_embeddings.shape)

        reward = self._compute_reward(pred_embeddings, gt_embeddings)       

        return reward


    def step(self, action):
        # print('self.current_user_index ',self.current_user_index)
        # print('self.current_sequence_index ',self.current_sequence_index)
        # print('Step No: '+str(self.current_user_index)+'.'+str(self.current_sequence_index))
        
        # print('Action: ', action) 
        
        state = self._get_current_state()
        reward = self.get_rewards(action, state)
        self.rewards_log.append(reward)
        
        # print('Reward: ', reward)
        
        if self.current_sequence_index < len(self.recommender.state_sequences[self.current_user_index]) - 1:
            self.current_sequence_index += 1
        else:
            self.current_user_index += 1
            self.current_sequence_index = 0
            
        if self.current_user_index >= len(self.recommender.users_map):
            print("---------------------End of Episode------------------------")
            self.done = True # End of episode       
            rwd_df = pd.DataFrame(self.rewards_log)
            rwd_df.to_csv('data/log/rewards_log.csv')

        return state, reward, self.done, {}

    def reset(self):
        self.current_user_index = 0
        self.current_sequence_index = 0
        self.done = False
        return self._get_current_state()

    def _get_current_state(self):
        state = State(self.current_user_index, self.current_sequence_index, self.recommender, self.max_item_len)
        return state.get_state()
    

    def render(self, mode='human'):
        # Optional for visualization
        pass

    def close(self):
        # Optional cleanup
        pass

    def seed(self, seed=None):
        np.random.seed(seed)


def make_recommender_env(num_env, rec):
    """
    Utility function for creating the vectorized environment.
 
    :param num_env: Number of parallel environments to create.
    :return: A vectorized Gym environment.
    """
    def _init():
        return RecommenderEnv(rec)
    return make_vec_env(_init, n_envs=num_env)


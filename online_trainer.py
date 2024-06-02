import numpy as np
import torch
from gym.spaces import Box

import warnings
warnings.filterwarnings("ignore")

from stable_baselines3.common.utils import obs_as_tensor
import numpy as np
import torch.nn.functional as F

class Online_Recommendation_Trainer:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.step_buffer = self.agent.rollout_buffer

    def dict_to_tensor(self, dict_obs, device):
        return {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device) for k, v in dict_obs.items()}
    
    def get_top_movies(self, state, actions):
            actions[(state['movie_ratings'] > 0) | (state['movie_ratings'] == -5.0)] = float('-inf')
            actions = torch.tensor(actions, dtype=torch.float32)                      
            _, top_indices = torch.topk(actions, 5)
            rec_movies = state['movie_indices'][top_indices]
        
            return rec_movies

    def online_update(self, num_steps=1):
        advantages = torch.zeros(1, device=self.agent.device)       
        state = self.env.get_current_state()   

        print(f"Current state: {state}")


        for key, value in state.items():
                if torch.isnan(torch.tensor(value)).any():
                    print(f"NaN detected in state {key}: {value}")
                    raise ValueError(f"NaN detected in state {key}")


        actions = self.agent.predict(state,deterministic=True)[0]   
        print(f"Predicted actions: {actions}")

        if torch.isnan(torch.tensor(actions)).any():
                print(f"NaN detected in actions: {actions}")
                raise ValueError("NaN detected in actions")        
                                                                                                    # Get the actions from the agent
        obs_tensor = self.dict_to_tensor(state, self.agent.device)          
        
                                                                                                                                    # Convert the state to tensor
        with torch.no_grad():
            values = self.agent.policy.predict_values(obs_tensor)                                                                                                       # Get the values from the policy
            log_probs = self.agent.policy.evaluate_actions(obs_tensor, torch.tensor([actions]).to(self.agent.device))[1] 
                                                                                                    # Get the log probs from the policy     
        
        rec_movies = self.get_top_movies(state, actions)      
        
        
        print('rec_movies ::: ',rec_movies)    
                                                                      # Get the top 5 recommended movies
        next_state, reward, done, _ = self.env.step(rec_movies)  
        
        print(f"Reward: {reward}, Done: {done}")
                                                                                        # Take a step in the environment
        # Convert done to float
        #done_tensor = torch.tensor([done], dtype=torch.float32, device=self.agent.device)
        #done_tensor = torch.tensor([float(done)], dtype=torch.float32, device=self.agent.device)
        done_float = float(done)
        done_numpy = np.array([done_float], dtype=np.float32)

                                                                    
        self.step_buffer.buffer_size = 1                                                                                   # Set the buffer size to 1
    
        self.step_buffer.add(obs_tensor, actions, reward, done_numpy, values, log_probs)                                            # Add the data to the buffer
        
        next_obs_tensor = self.dict_to_tensor(next_state, self.agent.device)    


        #print('self.step_buffer ',self.step_buffer) 
        
        
                                                                                                                            # Convert the next state to tensor
        with torch.no_grad():
            next_values = self.agent.policy.predict_values(next_obs_tensor)  
                                              # Get the values of the next state
        self.step_buffer.compute_returns_and_advantage(next_values, done_numpy)                                             # Compute the returns and advantages
        self.agent.train()                                                                                        # Train the agent
        self.step_buffer.reset()                                                                           # Reset the buffer
        

        # # Perform a single update using the buffer
        # # rollout_data = self.step_buffer.get(1)
        # self.agent.train()  # Set the model to training mode
        # # self.agent._update(rollout_data, 1)  # Update the model with one mini-batch   
        
        #---------------------------------------
        
        # # Manually collect rollout data
        # rollout_data = {
        #     "observations": self.env.get_current_state(),
        #     "actions": torch.tensor(actions, dtype=torch.float32).to(self.agent.device),
        #     "rewards": torch.tensor(reward, dtype=torch.float32).to(self.agent.device),
        #     "dones": torch.tensor(done, dtype=torch.float32).to(self.agent.device),
        #     "values": values,
        #     "log_probs": log_probs
        # }

        # # Compute returns and advantages for the single step
        # with torch.no_grad():
        #     next_obs_tensor = self.dict_to_tensor(next_state, self.agent.device)
        #     next_values = self.agent.policy.predict_values(next_obs_tensor)

        # returns = rollout_data["rewards"] + self.agent.gamma * next_values * (1 - rollout_data["dones"])
        # advantages = returns - rollout_data["values"]
        
        
        # delta = rollout_data["rewards"] + self.agent.gamma * next_values * (1 - rollout_data["dones"]) - values
        # advantages = delta + self.agent.gamma * self.agent.gae_lambda * advantages

        # returns = values + advantages

        # # Compute loss
        # ratio = torch.exp(rollout_data["log_probs"] - rollout_data["log_probs"].detach())
        # surr1 = ratio * advantages
        # surr2 = torch.clamp(ratio, 1 - self.agent.policy_clip_range, 1 + self.agent.policy_clip_range) * advantages
        # policy_loss = -torch.min(surr1, surr2).mean()

        # value_loss = F.mse_loss(rollout_data["values"], returns)

        # # Update policy
        # self.agent.policy.optimizer.zero_grad()
        # loss = policy_loss + self.agent.value_loss_coef * value_loss
        # loss.backward()
        # self.agent.policy.optimizer.step()
            
        #---------------------------------------
    
        # # Train the model using the manually collected rollout data
        # self.agent.policy.optimizer.zero_grad()
        # # loss = self.agent.policy.loss(
        # #     rollout_data["observations"],
        # #     rollout_data["actions"],
        # #     rollout_data["log_probs"],
        # #     returns,
        # #     advantages
        # # )
        # # loss.backward()
        # self.agent.policy.optimizer.step()
        
        self.env.state = next_state
            
        return rec_movies
    

    def recommend_movies(self):
        state = self.env.get_current_state()   
        actions = self.agent.predict(state, deterministic=True)[0]  
        rec_movies = self.get_top_movies(state, actions) 
        return rec_movies



# class Online_Recommendation_Trainer:
#     def __init__(self, agent, env):
#         self.agent = agent
#         self.env = env
#         self.step_buffer = self.agent.rollout_buffer

#     def dict_to_tensor(self, dict_obs, device):
#         # Convert dict observation to dict of tensors
#         return {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device) for k, v in dict_obs.items()}

#     def online_update(self, num_steps=1):
#         for _ in range(num_steps):
#             state = self.env.get_current_state()
#             actions = self.agent.predict(state, deterministic=True)[0]

#             obs_tensor = self.dict_to_tensor(state, self.agent.device)
#             with torch.no_grad():
#                 values = self.agent.policy.predict_values(obs_tensor)
#                 log_probs = self.agent.policy.evaluate_actions(obs_tensor, torch.tensor([actions]).to(self.agent.device))[1]       
            
#             actions[(state['movie_ratings'] > 0) | (state['movie_ratings'] == -5.0)] = float('-inf')      # Invalidate scores for visited movies
#             actions = torch.tensor(actions, dtype=torch.float32)                      
#             _, top_indices = torch.topk(actions, 5)                                  # Get top-5 movie indices
#             rec_movies = state['movie_indices'][top_indices]
#             print("Recommended Movies", rec_movies)
            
#             next_state, reward, done, _ = self.env.step(rec_movies)

#             self.step_buffer.add(self.env.get_current_state(), actions, reward, done, values, log_probs)

#             next_obs_tensor = self.dict_to_tensor(next_state, self.agent.device)
#             with torch.no_grad():
#                 next_values = self.agent.policy.predict_values(next_obs_tensor)
#             self.step_buffer.compute_returns_and_advantage(next_values, done)
#             self.agent.train()
#             self.step_buffer.reset()
            
#             # if len(self.step_buffer) == self.step_buffer.buffer_size:
#             #     self.step_buffer.compute_returns_and_advantage(last_values=self.agent.policy.predict_values(next_state))
#             #     self.agent.train()
#             #     self.step_buffer.reset()
            

#             self.env.state = next_state
#         return rec_movies

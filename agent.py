import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from gym import spaces


class ItemEmbeddingAdapter(nn.Module):
    def __init__(self, input_channels, output_channels, keep_dim=True):
        super(ItemEmbeddingAdapter, self).__init__()
        # Define the convolutional layer
        if keep_dim:
            # Use padding to maintain dimensionality
            self.conv1d = nn.Conv1d(in_channels=input_channels,
                                    out_channels=output_channels,
                                    kernel_size=3,  # Example kernel size
                                    padding=1)  # Padding to keep the dimension the same
        else:
            # No padding, dimensionality will reduce
            self.conv1d = nn.Conv1d(in_channels=input_channels,
                                    out_channels=output_channels,
                                    kernel_size=3)

    def forward(self, x):
        # x shape expected: [batch, channels, length] -> [64, 100, 185]
        x = x.permute(0, 2, 1)  # Rearrange to [batch, length, channels]
        x = self.conv1d(x)  # Apply convolution
        x = F.relu(x)  # Apply non-linear activation
        x = x.permute(0, 2, 1)  # Rearrange back to [batch, channels, length] -> [64, 185, 32]
        return x
    
class RecommendationPolicyCNN(nn.Module):

    def __init__(self, num_users, num_movies, user_embedding_dim,
                 movie_embedding_dim, adapted_movie_embedding_dim, mlp_dims, conv_out_channels, kernel_size):
        super(RecommendationPolicyCNN, self).__init__()

        self.num_users = num_users
        self.num_movies = num_movies

        self.user_embedding_dim = user_embedding_dim
        self.movie_embedding_dim = movie_embedding_dim

        # User Embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, user_embedding_dim)
        self.user_embedding_mlp = nn.Embedding(num_users, user_embedding_dim)

        self.item_embeddingadapter=ItemEmbeddingAdapter(input_channels=movie_embedding_dim,
                                                        output_channels=adapted_movie_embedding_dim)

        # GMF
        self.gmf_fc = nn.Linear(adapted_movie_embedding_dim, user_embedding_dim)

        # print('user_embedding_dim ',user_embedding_dim)
        # print('movie_embedding_dim ',adapted_movie_embedding_dim)

        mlp_input_dim = user_embedding_dim + adapted_movie_embedding_dim  # Concatenated dimension for user and movie embeddings

        self.mlp_fc_layers = nn.ModuleList([nn.Linear(mlp_input_dim, mlp_dims[0])])

        for in_size, out_size in zip(mlp_dims[:-1], mlp_dims[1:]):
            self.mlp_fc_layers.append(nn.Linear(in_size, out_size))

        # Final prediction layer outputs a score for each movie
        self.output_fc = nn.Linear(user_embedding_dim + mlp_dims[-1], 1)


    def forward(self, user_ids, movie_ids, movie_embeddings, ratings):

        user_ids = user_ids.long()

        user_embed_gmf = self.user_embedding_gmf(user_ids)  # Shape: [batch_size, user_embedding_dim]
        # print('user_embed_gmf ',user_embed_gmf.shape)

        user_embed_mlp = self.user_embedding_mlp(user_ids)
        # print('user_embed_mlp ',user_embed_mlp.shape)

        adapted_embeddings =  self.item_embeddingadapter(movie_embeddings)
        # print('adapted_embeddings ',adapted_embeddings.shape)
        
        # GMF path
        gmf_out = self.gmf_fc(adapted_embeddings)  # Processing movie embeddings directly
        gmf_out = gmf_out * user_embed_gmf
        # print('movie_embeddings shape',adapted_embeddings.shape)

        # MLP path
        mlp_input = torch.cat((user_embed_gmf, adapted_embeddings), dim=2)  # Concatenate user and movie embeddings directly
        mlp_out = mlp_input
        for layer in self.mlp_fc_layers:
            mlp_out = F.relu(layer(mlp_out))

        # Combine GMF and MLP outputs

        # print('gmf_out ',gmf_out.shape)
        # print('mlp_out ',mlp_out.shape)

        combined_features = torch.cat((gmf_out, mlp_out), dim=2)
        output = self.output_fc(combined_features)
        # print('output ',output.shape)   
        output = output.squeeze(-1)  # Ensure the output shape is correct
        
        return output
    
    
class CustomRecommendationFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that adapts the input observation space to the inputs expected by RecommendationPolicyCNN.
    """
    def __init__(self, observation_space: spaces.Dict,  
                 num_users=1000, num_movies=185, user_embedding_dim=16, 
                 movie_embedding_dim=100, adapted_movie_embedding_dim=32,
                 mlp_dims=[512,256,128], conv_out_channels=128, kernel_size=3):

        super().__init__(observation_space, features_dim=185) 

        self.recommendation_policy_cnn = RecommendationPolicyCNN(
            num_users=num_users,
            num_movies=num_movies,
            user_embedding_dim=user_embedding_dim,
            movie_embedding_dim=movie_embedding_dim,
            adapted_movie_embedding_dim=adapted_movie_embedding_dim,
            mlp_dims=mlp_dims,
            conv_out_channels=conv_out_channels,
            kernel_size=kernel_size
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        user_indices = observations["user_indices"].clone().detach()
        # print("User Indices Shape", user_indices.shape)
        # movie_embeddings = torch.tensor(observations["movie_embeddings"],dtype=torch.float32)
        movie_embeddings = observations["movie_embeddings"].clone().detach()
        # print("Movie Embeddings Pre-Shape", movie_embeddings.shape)
        movie_ratings = observations["movie_ratings"].clone().detach()
        movie_indices = observations["movie_indices"].clone().detach()
        # print("Movie Indices Shape", movie_indices.shape)
        # print("Movie Ratings Shape", movie_ratings.shape)
        # print("Movie Embeddings Shape", movie_embeddings.shape)
        features = self.recommendation_policy_cnn(user_indices, movie_indices,movie_embeddings, movie_ratings)
        # print("Features Shape", features.shape)
        return features
    

class CustomActorCriticPolicy(ActorCriticPolicy):
    
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Space, 
                 lr_schedule, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, 
                                                      features_extractor_class=CustomRecommendationFeaturesExtractor, 
                                                      net_arch=dict(pi=[185,185], vf=[185,185]),
                                                      **kwargs)

        self.action_net = nn.Sequential(
            nn.Linear(self.features_extractor.features_dim, action_space.shape[0]),
            nn.Softmax(dim=-1)
        )

        self.value_net = nn.Linear(self.features_extractor.features_dim, 1)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(obs)

        return self.action_net(features), self.value_net(features), torch.log(torch.tensor([0.1]))

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.features_extractor(obs)
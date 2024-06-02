import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gym import spaces

class ItemEmbeddingAdapter(nn.Module):
    def __init__(self, input_channels, output_channels, keep_dim=True):
        super(ItemEmbeddingAdapter, self).__init__()
        if keep_dim:
            self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1)
        else:
            self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=3)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        return x

class RecommendationPolicyHybrid(nn.Module):
    def __init__(self, num_users, num_movies, user_embedding_dim, movie_embedding_dim, adapted_movie_embedding_dim, nmf_latent_dim, mlp_dims, conv_out_channels, kernel_size):
        super(RecommendationPolicyHybrid, self).__init__()

        self.num_users = num_users
        self.num_movies = num_movies

        self.user_embedding_dim = user_embedding_dim
        self.movie_embedding_dim = movie_embedding_dim

        # Movie Embeddings Adapter
        self.item_embedding_adapter = ItemEmbeddingAdapter(input_channels=movie_embedding_dim, output_channels=adapted_movie_embedding_dim)

        # NMF Latent Factors
        self.nmf_user_factors = nn.Embedding(num_users, nmf_latent_dim)
        self.nmf_item_factors = nn.Embedding(num_movies, nmf_latent_dim)

        # MLP layers for combining user, movie embeddings and NMF factors
        mlp_input_dim = adapted_movie_embedding_dim + nmf_latent_dim + 1  # +1 for ratings

        self.mlp_fc_layers = nn.ModuleList([nn.Linear(mlp_input_dim, mlp_dims[0])])
        for in_size, out_size in zip(mlp_dims[:-1], mlp_dims[1:]):
            self.mlp_fc_layers.append(nn.Linear(in_size, out_size))

        # Output layer to predict ranking score
        self.output_fc = nn.Linear(mlp_dims[-1], 1)

    def forward(self, user_ids, movie_ids, movie_embeddings, ratings):
        user_ids = user_ids.long()
        movie_ids = movie_ids.long()

        # Adapt movie embeddings
        adapted_embeddings = self.item_embedding_adapter(movie_embeddings)
        adapted_embeddings = F.relu(adapted_embeddings)  # Ensure non-negativity

        # Get NMF latent factors
        nmf_user_factor = self.nmf_user_factors(user_ids)
        nmf_item_factor = self.nmf_item_factors(movie_ids)

        # Concatenate adapted movie embeddings, NMF factors, and ratings
        ratings = ratings.unsqueeze(2)  # Ensure ratings are the correct shape
        combined_input = torch.cat((adapted_embeddings, nmf_user_factor, nmf_item_factor, ratings), dim=2)
        combined_input = F.relu(combined_input)  # Ensure non-negativity

        # Pass through MLP layers
        mlp_out = combined_input
        for layer in self.mlp_fc_layers:
            mlp_out = F.relu(layer(mlp_out))

        # Predict ranking scores
        output = self.output_fc(mlp_out)
        output = output.squeeze(-1)  # Ensure the output shape is correct

        return output

class CustomRecommendationFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, num_users=1000, num_movies=185, user_embedding_dim=16, movie_embedding_dim=100, adapted_movie_embedding_dim=32, nmf_latent_dim=20, mlp_dims=[512, 256, 128], conv_out_channels=128, kernel_size=3):
        super().__init__(observation_space, features_dim=adapted_movie_embedding_dim + nmf_latent_dim + 1)
        self.recommendation_policy_hybrid = RecommendationPolicyHybrid(
            num_users=num_users,
            num_movies=num_movies,
            movie_embedding_dim=movie_embedding_dim,
            adapted_movie_embedding_dim=adapted_movie_embedding_dim,
            nmf_latent_dim=nmf_latent_dim,
            mlp_dims=mlp_dims,
            conv_out_channels=conv_out_channels,
            kernel_size=kernel_size
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        user_indices = observations["user_indices"].clone().detach()
        movie_indices = observations["movie_indices"].clone().detach()
        movie_embeddings = observations["movie_embeddings"].clone().detach()
        movie_ratings = observations["movie_ratings"].clone().detach()
        features = self.recommendation_policy_hybrid(user_indices, movie_indices, movie_embeddings, movie_ratings)
        return features

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Space, lr_schedule, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, features_extractor_class=CustomRecommendationFeaturesExtractor, net_arch=dict(pi=[512, 256], vf=[512, 256]), **kwargs)
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

    def get_distribution(self, obs: torch.Tensor):
        features = self.extract_features(obs)
        mean_actions = self.action_net(features)
        return mean_actions

    def _get_latent(self, obs: torch.Tensor):
        features = self.extract_features(obs)
        return features, features, None

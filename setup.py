import numpy as np
import pandas as pd
import random

class User:
    """
    User object to hold the user features.
    """
    def __init__(self, userId, encodedUserId, movieRating):
        self.userId = userId
        self.encodedUserId = encodedUserId
        self.movieRating = movieRating
    
    def get_user_rating(self, movie_id, watch_list):
        for movieid, rating in self.movieRating:
            if movie_id == movieid & movie_id in watch_list:
                return rating
        return 0.0
    
    def to_dict(self):
        return {'userId': self.userId, 'encodedUserId': self.encodedUserId, 'movieRating': self.movieRating}
    
       
 
class Movie:
    """
    Movie object to hold the movie features.
    """
    def __init__(self, movieId, encodedMovieId, title, genre, tag, desc, descEmbedding):
        self.movieId = movieId
        self.encodedMovieId = encodedMovieId
        self.title = title
        self.genre = genre
        self.tag = tag
        self.desc = desc
        self.descEmbedding = descEmbedding
        
    def to_dict(self):
        return {'movieId': self.movieId, 'encodedMovieId': self.encodedMovieId, 'title': self.title, 'genre': self.genre, 'tag': self.tag, 'desc': self.desc, 'descEmbedding': self.descEmbedding}
        
        
 
class Recommender():
    """
    Recommender object to load the movie and ratings features and create the user and movie objects.
    """
    def __init__(self, movie_features_path,ratings_features_path):
        super(Recommender, self).__init__()
        self.movies_data, self.users_data, self.ratings_data, self.test_users, self.test_ratings = self.load_features(movie_features_path, ratings_features_path)
       
        self.movies_map, self.users_map = self.get_list_of_objects(self.movies_data, self.users_data)
        # self.test_movies_map, self.test_users_map = self.get_list_of_objects(self.movies_data, self.test_users)
               
        self.sequences = self.get_sequences(self.movies_data,self.ratings_data) 
        # self.test_sequences = self.get_sequences(self.movies_data,self.test_ratings)

        self.state_sequences = self.get_user_sequence_combinations()
       
 
    def load_features(self,movie_features_path,ratings_features_path):
        movies = pd.read_csv(movie_features_path).drop(['Unnamed: 0'], axis=1)
        ratings = pd.read_csv(ratings_features_path).drop(['Unnamed: 0','timestamp'], axis=1)
        
        train_users = ratings['userId'].unique()[:1000] # Selecting only 1000 users for training
        train_ratings = ratings[ratings['userId'].isin(train_users)] # Select from ratings only of train users
        train_users = train_ratings.groupby(['userId','encoded_user_ids']).apply(lambda x: list(zip(x['movieId'], x['rating']))).reset_index()

        train_users.rename(columns = {0:'movie_rating'}, inplace = True)
        
        # test_users = ratings['userId'].unique()[1000:1050] # Selecting only 50 users for testing
        test_users = ratings['userId'].unique()[1000:1500]
        test_ratings = ratings[ratings['userId'].isin(test_users)] # Select from ratings only of test users
        test_users = test_ratings.groupby(['userId','encoded_user_ids']).apply(lambda x: list(zip(x['movieId'], x['rating']))).reset_index()
        
        test_users.rename(columns = {0:'movie_rating'}, inplace = True)
 
        return movies, train_users, train_ratings, test_users, test_ratings
   
    def get_list_of_objects(self,movies_data, user_data):
        users_map = {}
        movies_map = {}

        for row in user_data.itertuples():
            user_id = row.userId
            encoded_user_id = row.encoded_user_ids
            movie_rating = row.movie_rating
            user = User(user_id, encoded_user_id, movie_rating)
            users_map[user_id]=user
 
        for row in movies_data.itertuples():
            movie_id = row.movieId
            encoded_movie_id = row.encoded_movie_ids
            title = row.title
            genre = row.genres
            tags = row.tag
            desc = row.desc
            desc_embedding = np.array(row.desc_embedding_glove.replace('\n', '').replace('[', '').replace(']', '').split(), dtype=np.float64)
            movie = Movie(movie_id, encoded_movie_id, title, genre, tags, desc, desc_embedding)
            movies_map[movie_id] = movie
                
        return movies_map, users_map
    
 
    def get_sequences(self,moviesdf, ratingsdf):

        all_movies = moviesdf.movieId.unique().tolist()
        all_users = ratingsdf.userId.unique().tolist()
 
        user_objects = []
   
        for user_id in all_users:
            user_ratings = ratingsdf[ratingsdf['userId'] == user_id]
            user_details = {
                'user_id': user_id,                
                'ratings_df': user_ratings
            }
            user_movies_watched = user_ratings['movieId'].tolist()
            user_movie_objects = []
            for movie_id in user_movies_watched:
                movie_details = moviesdf[moviesdf['movieId'] == movie_id]
                movie_object = {
                    'movie_id': movie_id,
                    'genre': movie_details['genres'].iloc[0],
                    'tag': movie_details['tag'].iloc[0],
                    'title': movie_details['title'].iloc[0],
                    'desc': movie_details['desc'].iloc[0],
                    'desc_embedding': movie_details['desc_embedding_glove'].iloc[0]
                }
                user_movie_objects.append(movie_object)
            user_objects.append({
                'user_details': user_details,
                'movie_objects': user_movie_objects
            })

        sequence_dict = {}
       
        for user_obj in user_objects:
            user_id = user_obj['user_details']['user_id']
            user_ratings = user_obj['user_details']['ratings_df']
            movie_seq = user_ratings['movieId'].tolist()
            n = 5
            m = 5
            k = 2
            sequence_lst = []
            while n + m <= len(movie_seq):
                movie_watched_bucket = movie_seq[:n]
                movie_to_be_watched_bucket = movie_seq[n:]
                common_pos_samples = list(set(movie_watched_bucket).union(set(movie_to_be_watched_bucket)))
                total_neg_samples = [mv for mv in all_movies if mv not in movie_seq]
                selected_mix_samples = total_neg_samples + common_pos_samples
                selected_mix_samples = sorted(selected_mix_samples) # Sort the selected_mix_samples
                # random.shuffle(selected_mix_samples)
                sequence_lst.append({'pos_samples': movie_watched_bucket,
                                     'mix_samples': selected_mix_samples,
                                     'next_pos_samples': movie_to_be_watched_bucket})
                n = n + m
            sequence_dict[user_id] = sequence_lst
 
        df = pd.DataFrame(sequence_dict.items(), columns=['user_id', 'sequences'])

        df.to_csv('data/sequences/sequences.csv')
        return df
 
    def get_user_sequence_combinations(self):
       
        user_seq_lst = {}
        for indx, row in self.sequences.iterrows():    
            user_id = row['user_id']
            user_ob = self.users_map[user_id]
            sequence_lst = row['sequences']
            row_lst = []
            for seq in sequence_lst:
                seq_dict = {}
                pos_neg_samples = seq['mix_samples']  
                pos_samples = seq['pos_samples']    # movie_watched_bucket
                seq_dict['movie_indices'] = np.array([self.movies_map[movie_id].encodedMovieId for movie_id in  pos_neg_samples],dtype="float")
                seq_dict['movie_embeddings'] = np.array([self.movies_map[movie_id].descEmbedding for movie_id in pos_neg_samples],dtype="float")
                # seq_dict['movie_ratings'] =  np.array([user_ob.get_user_rating(movie_id) for movie_id in pos_neg_samples],dtype="float")
                seq_dict['movie_ratings'] =  np.array([user_ob.get_user_rating(movie_id, pos_samples) for movie_id in pos_neg_samples],dtype="float")
                seq_dict['user_indices'] = np.array([user_ob.encodedUserId for movie_id in  pos_neg_samples],dtype="float")
                row_lst.append(seq_dict)
            user_seq_lst[user_ob.encodedUserId] = row_lst
            
        user_seq = pd.DataFrame(user_seq_lst.items(), columns=['user_id', 'sequences'])
        user_seq.to_csv('data/sequences/user_sequences.csv')
       
        return user_seq_lst
    
    def get_test_users(self, num_users=50):
        _, self.test_users_map = self.get_list_of_objects(self.movies_data, self.test_users)
        if num_users >= len(self.test_users_map):
            self.test_users_map = self.test_users_map
        else:
            self.test_users_map = random.sample(list(self.test_users_map.items()), num_users)
        test_users={}
        for user, user_object in self.test_users_map:
            test_users[user] = user_object.to_dict()
        
        return test_users
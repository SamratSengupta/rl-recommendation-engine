import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from inference import recommend_movies

def evaluate_model(model, rec):
    """
    Evaluate the model using NDCG metric with random 50 users.
    """
    
    def movieid_to_index(movieid):
        return rec.movies_map[movieid].encodedMovieId 

    def get_gt_movies(user_id):
        return [movieid_to_index(movieid) for movieid in rec.sequences['sequences'][user_id][1]['next_pos_samples']]

    def movieindex_to_id(movie_index):
        for key, movie in rec.movies_map.items():
            if movie.encodedMovieId == movie_index:
                return movie.movieId

    def compute_ndcg(recMovies, gtMovies, rec):
        
        pred_embeddings = np.array([rec.movies_map[movieindex_to_id(movie_id)].descEmbedding for movie_id in recMovies])
        gt_embeddings = np.array([rec.movies_map[movieindex_to_id(movie_id)].descEmbedding for movie_id in gtMovies])

        ndcg_scores = []
        for n, predicted_embedding in enumerate(pred_embeddings):
            similarities = cosine_similarity([predicted_embedding], gt_embeddings)[0]
            max_similarity = np.max(similarities)
            current_similarity = similarities[n]
            # print(max_similarity, current_similarity)
        
            # Rank the ground truth movies based on cosine similarity
            ranked_indices = np.argsort(similarities)[::-1][:5]  # Top 5 similar movies
            ranked_movies = [gtMovies[i] for i in ranked_indices]
            
            # Compute DCG for the ranked movies
            dcg = 0
            for i, movie in enumerate(ranked_movies):
                dcg += (2 ** max_similarity - 1) / np.log2(i + 2)  # Discounted gain
            
            # Compute ideal DCG for perfect ranking
            idcg = sum((2 ** 1 - 1) / np.log2(i + 2) for i in range(min(5, len(gtMovies))))
            
            # Compute NDCG
            if idcg == 0:
                ndcg = 0  # Handle division by zero
            else:
                ndcg = dcg / idcg
            ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores)

    all_states = []
    for uid, seq in rec.state_sequences.items():
        all_states.append(seq[1])

    # eval_states = random.sample(all_states, 50)
            
    all_data = []
    for state in all_states:
        user_id = int(state['user_indices'][0])
        user_watched_movies = [int(movie) for movie, rating in zip(state['movie_indices'], state['movie_ratings']) if rating > 0]
        user_watched_ratings = [rating for rating in state['movie_ratings'] if rating > 0]
        all_data.append((user_id, user_watched_movies, user_watched_ratings))
        
    eval_data = random.sample(all_data, 50)

    # Create df
    eval_df = pd.DataFrame(eval_data, columns=['user_id', 'user_watched_movies', 'user_watched_ratings'])
    eval_df = eval_df.set_index('user_id')
    # eval_df["user_watched_movies"] = eval_df["user_watched_movies"].apply(lambda x: [movieid_to_index(movieid) for movieid in x])

    # Evaluate the model with inference
    eval_users_data = eval_df.T.to_dict('list')
    eval_df['recMovies'] = eval_df['user_watched_movies'].apply(lambda x: [])
    for user_id, user_data in eval_users_data.items():
        user_watched_movies = user_data[0].copy()
        user_watched_ratings = user_data[1].copy()
        recommended_movies = recommend_movies(model, user_id, user_watched_movies, user_watched_ratings, rec)
        eval_df.at[user_id, 'recMovies'] = [movieid_to_index(movieid) for movieid in recommended_movies]
        
    eval_df['gtMovies'] = eval_df.index.map(get_gt_movies)
    # eval_df['precision'] = eval_df.apply(lambda x: len(set(x['recMovies']).intersection(set(x['gtMovies']))) / len(set(x['recMovies'])), axis=1) 

    eval_df['ndcg'] = eval_df.apply(lambda x: compute_ndcg(x['recMovies'], x['gtMovies'], rec), axis=1)
    
    return eval_df


def test_recommender(model, rec):
    """
    Test the recommender model with random 10 users.
    """
    test_users = rec.get_test_users(10)
    test_df = pd.DataFrame(test_users).T.drop(columns=['userId'])
    test_df.columns = ['userIndex', 'movieRating']
    test_df.reset_index(drop=True, inplace=True)
    test_df['watchedMoviesId'] = test_df['movieRating'].apply(lambda x: [movie for movie, rating in x][0:10])
    test_df['watchedMoviesRating'] = test_df['movieRating'].apply(lambda x: [rating for movie, rating in x][0:10])
    test_df = test_df.drop(columns=['movieRating'])
    
    test_df['watchedMoviesObjects'] = test_df['watchedMoviesId'].apply(lambda x: [rec.movies_map[movie] for movie in x])

    eval_users_data = test_df.T.to_dict('list')
    test_df['recMovies'] = test_df['watchedMoviesId'].apply(lambda x: [])
    for index, user_data in eval_users_data.items():
        user_id = user_data[0]
        user_watched_movies = user_data[1].copy()
        user_watched_ratings = user_data[2].copy()
        recommended_movies = recommend_movies(model, user_id, user_watched_movies, user_watched_ratings, rec)
        test_df.at[index, 'recMovies'] = recommended_movies
    
    test_df['recMoviesObjects'] = test_df['recMovies'].apply(lambda x: [rec.movies_map[movie] for movie in x])
    
    def get_movie_desc(movielist):
        desclist = []
        for movie in movielist:
            title = movie.title
            genre = movie.genre
            tags = movie.tag
            des = f"{title} | {genre}"
            tag = f"{tags}"
            desc = {"desc" : des, "tag" : tag }
            desclist.append(desc)
        return desclist
    
    test_df['watchedMoviesDesc'] = test_df['watchedMoviesObjects'].apply(get_movie_desc)
    test_df['recMoviesDesc'] = test_df['recMoviesObjects'].apply(get_movie_desc)
        
    return test_df

def view_all_movies_from_test(test_df):
    output = test_df[['userIndex','watchedMoviesDesc','recMoviesDesc']].T.to_dict()
    for i in range(len(output)):
        print("-----------------------------------------------------------------------------------------")
        print(f"\033[1;33mUSER INDEX {output[i]['userIndex']}\033[0m")
        print("-----------------------------------------------------------------------------------------")
        print(f"\033[1;33mUser Watched Movies: \033[0m")
        for movieDesc in output[i]['watchedMoviesDesc']:
            print(f"{movieDesc['desc']} \033[34m{movieDesc['tag']}\033[0m")

        print(f"\033[1;33mOur Recommended Movies: \033[0m")
        for movieDesc in output[i]['recMoviesDesc']:
            print(f"{movieDesc['desc']} | \033[34m{movieDesc['tag']}\033[0m")
        
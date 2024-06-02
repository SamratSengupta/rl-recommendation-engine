import numpy as np

class State:
    """
    State class to hold the state of the environment. Returns observations form user_id and sequence_id.
    """
    def __init__(self, user_id, seq_id, rec, total_items):
        self.user_index = user_id
        self.sequence_index = seq_id
        
        self.userIndexes = np.array(rec.state_sequences[user_id][seq_id]['user_indices'])
        self.movieIndexes = np.array(rec.state_sequences[user_id][seq_id]['movie_indices'])
        self.movieEmbeddings = np.array(rec.state_sequences[user_id][seq_id]['movie_embeddings'])
        self.movieRatings = np.array(rec.state_sequences[user_id][seq_id]['movie_ratings'])   
        
    def get_state(self):
        observations = { "user_indices":self.userIndexes, "movie_indices":self.movieIndexes, "movie_embeddings":self.movieEmbeddings, 
                        "movie_ratings":self.movieRatings}
        return observations
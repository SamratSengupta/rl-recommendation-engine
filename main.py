import streamlit as st
from online_service import RecommendationService
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Movie Recommender", layout="wide")

# Initialize session state variables
for key, default in {'logged_in': False, 'flag': False, 'feedback': {'recommended_movies': {}, 'selected_movies': [], 'user_unwatched_movies': []}, 'start_index_rec': 0, 'start_index_unwatched': 0, 'start_index_watched': 0}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def get_recommendations_from_server(user_info):
    request_type = "GET_NEXT_RECOMMENDATIONS" if st.session_state.flag else "GET_FIRST_RECOMMENDATIONS"
    if request_type == "GET_FIRST_RECOMMENDATIONS":
        st.session_state.recommender = RecommendationService(user_info["user_id"])
    recommendations = st.session_state.recommender.get_recommended_movies()
    return recommendations

# Function to send feedback to the model
def send_feedback_to_model(feedback):
    st.session_state.recommender.submit_feedback(feedback)
    
def login():
    st.session_state.logged_in = True

def logout():
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.watched_movie_ids = []
    st.session_state.unwatched_movie_ids = []
    st.session_state.ratings = []
    st.session_state.flag = False
    st.session_state.feedback = {'recommended_movies': {}, 'selected_movies': [], 'user_unwatched_movies': []}

def display_movies(recommendations):
    feedback = st.session_state.feedback
    start_index_rec = st.session_state.start_index_rec
    start_index_unwatched = st.session_state.start_index_unwatched
    start_index_watched = st.session_state.start_index_watched
    
    
    recommended_movies = [st.session_state.recommender.env.movies_data.iloc[movie].to_dict() for movie in recommendations]
    # print(recommended_movies)
    rec_list = []
    for movie in recommended_movies:
        rec_list.append({
            "index": movie['movie_index'],
            "title": movie['movie_title'],
            "genre": movie['movie_genre']
        })
        
    num_recommendations = len(rec_list)
    end_index = min(start_index_rec + 3, num_recommendations)
    display_recommendations = rec_list[start_index_rec:end_index]
    st.subheader(f"Recommended Movies for You : [{len(recommendations)}]", anchor="recommendations", divider="blue")
    cols = st.columns(3)
    for i, movie in enumerate(display_recommendations):
        index = movie['index']
        title = movie['title']
        genre = movie['genre']
        container = cols[i].container(border=True, height=270)
        container.subheader(f'{title}', divider='blue')
        container.markdown(f'**Movie Index:** {index}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Genre:** {genre}')
        
        subCol1, subCol2 = container.columns(2)
        if subCol1.button("👍 Like", key=f"like_{index}"):
            feedback['recommended_movies'][index] = 1
        if subCol2.button("👎 Dislike", key=f"dislike_{index}"):
            feedback['recommended_movies'][index] = 0

        confirm = {1: '<span style="color:DarkKhaki">*You 👍 this movie!*</span>', 0: '<span style="color:DarkKhaki">*You 👎 this movie!*</span>'}.get(feedback['recommended_movies'].get(index, None), '')
        if confirm:
            container.markdown(confirm, unsafe_allow_html=True)

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 10, 1])
    if col1.button("Previous"):
        if start_index_rec > 0:
            st.session_state.start_index_rec = max(start_index_rec - 3, 0)
            st.rerun()
    if col3.button("View Next"):
        if end_index < len(recommendations):
            st.session_state.start_index_rec = min(start_index_rec + 3, len(recommendations) - 3)
            st.rerun()
  
  
    
    unwatched_movies = [st.session_state.recommender.env.movies_data.iloc[movie].to_dict() for movie in st.session_state.unwatched_movie_ids]
    # print(unwatched_movies)
    unwatched_list = []
    for movie in unwatched_movies:
        unwatched_list.append({
            "index": movie['movie_index'],
            "title": movie['movie_title'],
            "genre": movie['movie_genre']
        })
        
    num_unwatched = len(unwatched_list)
    end_index_unwanted = min(start_index_unwatched + 4, num_unwatched)
    display_unwatched = unwatched_list[start_index_unwatched:end_index_unwanted]
    st.subheader(f"Unwatched Movies : [{len(unwatched_list)}]", anchor="unwatched", divider="blue")
    cols = st.columns(4)
    for i, movie in enumerate(display_unwatched):
        index = movie['index']
        title = movie['title']
        genre = movie['genre']
        container = cols[i].container(border=True, height=230)
        container.subheader(f'{title}', divider='blue')
        container.markdown(f'**Movie Index:** {index}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Genre:** {genre}')
        
        SubCol1, SubCol2 = container.columns([1,1])
        selected_movie = SubCol1.checkbox("Interested", key=f"interested_{index}", value=(index in feedback['selected_movies']))
        if selected_movie or index in feedback['selected_movies']:
            if index not in feedback['selected_movies']:
                feedback['selected_movies'].append(index)
            SubCol2.markdown('<span style="color:DarkKhaki">*Added to watched!*</span>', unsafe_allow_html=True)
        
                
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 10, 1])
    if col1.button("Previous", key="unwatched-previous"):
        if start_index_unwatched > 0:
            st.session_state.start_index_unwatched = max(start_index_unwatched - 4, 0)
            st.rerun()
    if col3.button("View Next", key="unwatched-next"):
        if end_index_unwanted < len(st.session_state.unwatched_movie_ids):
            st.session_state.start_index_unwatched = min(start_index_unwatched + 4, len(st.session_state.unwatched_movie_ids) - 4)
            st.rerun()



    # st.session_state.watched_movie_ids.extend([movie for movie in recommendations])
    watched_movies = [st.session_state.recommender.env.movies_data.iloc[movie].to_dict() for movie in st.session_state.watched_movie_ids]
    # print(unwatched_movies)
    watched_list = []
    for movie in watched_movies:
        watched_list.append({
            "index": movie['movie_index'],
            "title": movie['movie_title'],
            "genre": movie['movie_genre']
        })
        
    num_watched = len(watched_list)
    end_index_wanted = min(start_index_watched + 4, num_watched)
    display_watched = watched_list[start_index_watched:end_index_wanted]
    st.subheader(f"Already watched movies : [{len(watched_list)}]", anchor="watched", divider="blue")
    cols = st.columns(4)
    for i, movie in enumerate(display_watched):
        index = movie['index']
        title = movie['title']
        genre = movie['genre']
        container = cols[i].container(border=True, height=200)
        # container.markdown(f'<h3 style="margin-bottom: 0;">{title}</h3><hr style="border: none; height: 1px; background-color: blue;">', unsafe_allow_html=True)
        container.subheader(f'{title}', divider='blue')
        container.markdown(f'**Movie Index:** {index}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Genre:** {genre}')
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 10, 1])
    if col1.button("Previous", key="watched-previous"):
        if start_index_watched > 0:
            st.session_state.start_index_watched = max(start_index_watched - 4, 0)
            st.rerun()
    if col3.button("View Next", key="watched-next"):
        if end_index_wanted < len(st.session_state.watched_movie_ids):
            st.session_state.start_index_watched = min(start_index_watched + 4, len(st.session_state.watched_movie_ids) - 4)
            st.rerun()
    
    feedback['user_unwatched_movies'] = st.session_state.unwatched_movie_ids
    
    return feedback

def login_page():

    def load_user_data(csv_file):
        df = pd.read_csv(csv_file)
        return df

    def authenticate_user(df, user_id, password):
        user = df[(df['user_id'] == user_id) & (df['password'] == password)]
        if not user.empty:
            return user.iloc[0]
        else:
            return None  
    _, col, _ = st.columns([2, 3, 2])
    col.title("Movie Recommender")

    user_id = col.text_input("User ID")
    password = col.text_input("Password", type="password")

    if col.button("Login"):
        if user_id and password:
            user_id = int(user_id)
            df = load_user_data('./data/log/users_data.csv')
            user = authenticate_user(df, user_id, password)
            
            if user is not None:
                st.toast("Login successful!")
                st.session_state.user_id = user_id
                st.session_state.watched_movie_ids = list(map(int, user['user_watched_movies'][1:-1].split(',')))
                st.session_state.ratings = list(map(float, user['user_watched_ratings'][1:-1].split(',')))
                st.session_state.unwatched_movie_ids = [m for m in range(0, 185) if m not in st.session_state.watched_movie_ids]
                login()
                # st.toast("Login successful!")
            else:
                st.toast("Invalid User ID or Password")
        else:
            st.toast("Please enter both User ID and Password")
        st.rerun()
        
        
def main_page():
    
    user_info = {
        "user_id": st.session_state.user_id,
        "watched_movie_ids": st.session_state.watched_movie_ids,
        "ratings": st.session_state.ratings
    }
    
    recommendations = get_recommendations_from_server(user_info)
    feedback = display_movies(recommendations)
    st.session_state.feedback = feedback
    
    sCol1, sCol2, sCol3, sCol4, sCol5 = st.columns([1, 2, 2, 2, 1])
    if sCol3.button("Submit", type="primary"):
        send_feedback_to_model(feedback)
        st.toast("Feedback submitted successfully!")
        print(feedback)
        st.session_state.watched_movie_ids = list(set(st.session_state.recommender.get_watched_movies()))
        st.session_state.ratings = st.session_state.recommender.env.user_data['user_watched_ratings']
        st.session_state.unwatched_movie_ids = list(set(st.session_state.recommender.get_unwatched_movies()))
        st.session_state.start_index_rec = 0
        st.session_state.start_index_unwatched = 0
        st.session_state.start_index_watched = 0
        
        st.session_state.flag = True
        st.session_state.feedback = {'recommended_movies': {}, 'selected_movies': [], 'user_unwatched_movies': []}
        st.rerun()

    if sCol4.button("Logout", type="primary"):
        st.toast("Recommendations stopped successfully!")
        logout()
        st.rerun()
    

if __name__ == "__main__":
    if st.session_state.logged_in:
        main_page()
    else:
        login_page()

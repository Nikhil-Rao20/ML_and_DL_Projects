import pickle
import pandas as pd
import streamlit as st
import requests


def fetch_poster(movie_id):
    response = requests.get('http://api.themoviedb.org/3/movie/{}?api_key=9fee289cbf3e65467480dacbead2703d&language=en-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500" + data['poster_path']


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        # fetching poster from API
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_movies_posters


movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Write down a movie to which we should recommend you a few?',
    movies['title'].values)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(posters[i])
            st.write("<h2 style='color:cyan;font-weight:bold;background-color:black;color:#e5e5e5;padding:10px; "
                     "font-size:1rem;text-shadow: #b2b2b2 2px 2px 3px;"
                     "font-family:monospace;border-radius:15px';display:flex;>{}</h2>".format(names[i]),
                     unsafe_allow_html=True)

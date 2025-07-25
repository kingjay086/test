import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\jay30\OneDrive\Documents\myprojects\python\imdb_top_1000.csv")  # Change this filename if using another dataset
    df = df[['Series_Title', 'Genre', 'Overview']]  # minimal required columns
    df.dropna(inplace=True)
    df['combined_features'] = df['Genre'] + " " + df['Overview']
    return df

# Generate recommendations
def recommend_movies(title, df, tfidf_matrix):
    try:
        idx = df[df['Series_Title'].str.lower() == title.lower()].index[0]
    except IndexError:
        return ["❌ Movie not found. Please check spelling."]
    
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 excluding itself

    movie_indices = [i[0] for i in sim_scores]
    return df['Series_Title'].iloc[movie_indices].tolist()

# Load data and vectorize
df = load_data()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Streamlit App UI
st.title("🎬 Movie Recommendation System")
st.write("Suggesting movies based on content (genres + plot).")

user_input = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    if user_input:
        recommendations = recommend_movies(user_input, df, tfidf_matrix)
        st.subheader("Top 5 Recommendations:")
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie}")
    else:
        st.warning("Please enter a movie title.")

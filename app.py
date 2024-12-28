from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

app = Flask(__name__)
CORS(app)

# Load model and MultiLabelBinarizer
model = joblib.load('model.joblib')
mlb = joblib.load('mlb.joblib')

# Load and merge data
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Merge movies and ratings
df = pd.merge(ratings_df, movies_df, on="movieId")

# Calculate average rating for each movie
movie_stats = df.groupby('movieId').agg({
    'rating': ['count', 'mean']
}).reset_index()
movie_stats.columns = ['movieId', 'rating_count', 'average_rating']

# Merge stats back to movies dataframe
movies_df = pd.merge(movies_df, movie_stats, on='movieId', how='left')
movies_df['average_rating'] = movies_df['average_rating'].fillna(0)
movies_df['rating_count'] = movies_df['rating_count'].fillna(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/movies', methods=['GET'])
def get_movies():
    # Returns a list of movie titles (for the form input)
    movies_list = movies_df[['movieId', 'title']].to_dict('records')
    return jsonify(movies_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        movie_title = data.get('movie_title')
        
        # Find the movie by title
        selected_movie = movies_df[movies_df['title'].str.contains(movie_title, case=False)].iloc[0]
        movie_id = selected_movie['movieId']
        
        # Get movie genres
        selected_genres = selected_movie['genres'].split('|')
        
        # Encode genres for all movies
        all_movies_genres = movies_df['genres'].str.split('|')
        genres_encoded = pd.DataFrame(
            mlb.transform(all_movies_genres),
            columns=mlb.classes_,
            index=movies_df.index
        )f
        
        # Get predictions
        probabilities = model.predict_proba(genres_encoded)[:, 1]
        
        # Create recommendations dataframe
        recommendations_df = movies_df.copy()
        recommendations_df['probability'] = probabilities
        
        # Filter out the selected movie and sort by probability
        recommendations_df = recommendations_df[recommendations_df['movieId'] != movie_id]
        recommendations_df = recommendations_df.sort_values('probability', ascending=False)
        
        # Get top 6 recommendations
        top_recommendations = recommendations_df.head(6)
        
        # Prepare response data
        recommendations_list = []
        for _, movie in top_recommendations.iterrows():
            recommendations_list.append({
                'movieId': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres'],
                'similarity_score': float(movie['probability']),
                'average_rating': float(movie['average_rating']),
                'rating_count': int(movie['rating_count'])
            })
        
        return jsonify({
            'recommendations': recommendations_list,
            'similarity_scores': [r['similarity_score'] for r in recommendations_list]
        })
        
    except Exception as e:
        print(f"Error in recommend route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

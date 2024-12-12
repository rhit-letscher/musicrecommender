import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import ast


#todo: get an input genre list from user's spotify playlist/song

def clean(df):
    # Convert string representation of list to actual list
    df['artist_genres'] = df['artist_genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])
    # Remove rows with empty genre lists
    cleaned_df = df[df['artist_genres'].apply(len) > 0]
    return cleaned_df

def vectorize_genre_list(genre_list, all_unique_genres):
    """
    Converts a list of genres into a weighted vector.
    """
    # Count how many times each genre appears
    genre_counts = Counter(genre_list)
    
    # Calculate total for normalization
    total_genres = max(len(genre_list), 1)  # Prevent division by zero
    
    # Create vector with a position for each possible genre
    genre_vector = np.zeros(len(all_unique_genres))
    
    # Fill in the normalized frequencies
    for i, genre in enumerate(all_unique_genres):
        genre_vector[i] = genre_counts.get(genre, 0) / total_genres
    
    return genre_vector

def extract_unique_genres(df):
    """
    Extracts all unique genres from the dataset.
    """
    all_genres = set()
    
    for genres_list in df['artist_genres']:
        if isinstance(genres_list, list):
            all_genres.update(genre.lower().strip() for genre in genres_list if genre)
    
    return sorted(list(all_genres))

def build_artist_genre_profiles(df, all_unique_genres):
    """
    Creates genre vectors for all artists in the dataset.
    """
    artist_profiles = {}
    
    for _, row in df.iterrows():
        artist_vector = vectorize_genre_list(row['artist_genres'], all_unique_genres)
        artist_profiles[row['artist_uri']] = artist_vector
    
    artist_vectors = np.array(list(artist_profiles.values()))
    return artist_profiles, artist_vectors

def recommend_artists(input_genres, artist_df, exclude_artists=None, n_recommendations=5):
    """
    Recommends artists based on input genres.
    """
    # Clean the data and get all unique genres
    cleaned_artist_df = clean(artist_df)
    all_unique_genres = extract_unique_genres(cleaned_artist_df)
    
    if not all_unique_genres:
        raise ValueError("No genres found in the dataset after cleaning")
    
    # Normalize input genres
    normalized_genres = [genre.lower().strip() for genre in input_genres]
    
    # Create genre vector from input genres
    input_vector = vectorize_genre_list(normalized_genres, all_unique_genres)
    
    # Create genre profiles for all artists
    artist_profiles, artist_vectors = build_artist_genre_profiles(cleaned_artist_df, all_unique_genres)
    
    if len(artist_profiles) == 0:
        raise ValueError("No valid artist profiles could be created")
    
    artists = list(artist_profiles.keys())
    
    # Set up and train KNN
    n_neighbors = min(len(artists), max(n_recommendations * 2, 5))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(artist_vectors)
    
    # Get recommendations
    distances, indices = knn.kneighbors([input_vector])
    
    # Filter recommendations
    exclude_artists = set() if exclude_artists is None else set(exclude_artists)
    recommended_artists = []
    recommendation_distances = []
    
    for idx, distance in zip(indices[0], distances[0]):
        artist = artists[idx]
        if artist not in exclude_artists:
            recommended_artists.append(artist)
            recommendation_distances.append(distance)
            if len(recommended_artists) >= n_recommendations:
                break
    
    return {
        'recommended_artists': recommended_artists[:n_recommendations],
        'distances': recommendation_distances[:n_recommendations],
        'all_genres': all_unique_genres
    }

def main():
    # Load the dataset
    df = pd.read_csv("artists.csv")
    
    # Create genre preferences
    genre_preferences = ['pop', 'pop', 'rock', 'hyperpop_italiano', 'hyperpop_italiano', 'indie pop', 'indie pop', 'uk bass', 'uk bass']
    
    try:
        recommendations = recommend_artists(genre_preferences, df)
        
        #results
        print("\nRecommended artists:")
        for artist, distance in zip(recommendations['recommended_artists'], 
                                  recommendations['distances']):
            print(f"Artist URI: {artist}, Distance: {distance:.3f}")

        #genre info   
        print(f"\nTotal unique genres: {len(recommendations['all_genres'])}")
        print(f"All genres: {recommendations['all_genres']}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nDataset preview:")
        print(df.head())
        print("\nColumns:", df.columns.tolist())

if __name__ == "__main__":
    main()
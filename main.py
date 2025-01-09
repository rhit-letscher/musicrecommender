import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import ast
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import dotenv

from api import *

dotenv.load_dotenv()



def clean(df):
    # Convert string representation of list to actual list
    df['artist_genres'] = df['artist_genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])
    # Remove rows with empty genre lists
    cleaned_df = df[df['artist_genres'].apply(len) > 0]
    return cleaned_df

def get_artist_names_from_uris(artist_uris):
    # Initialize Spotify client
    # You'll need to set up your credentials at https://developer.spotify.com/dashboard
    client_credentials_manager = SpotifyClientCredentials(
        client_id=os.environ.get('SPOTIFY_CLIENT_ID'),
        client_secret=os.environ.get('SPOTIFY_SECRET')
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Get artist names from URIs
    artist_names = []
    
    # Process URIs in batches of 50 (Spotify API limit)
    for i in range(0, len(artist_uris), 50):
        batch = artist_uris[i:i + 50]
        artists = sp.artists(batch)['artists']
        artist_names.extend([artist['name'] for artist in artists])
    
    return artist_names


def vectorize_genre_list(genre_list, all_unique_genres):
    """
    Converts a list of genres into a weighted vector.
    """
    assert len(genre_list) > 0
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
        all_genres.update(set(ast.literal_eval(genres_list)))

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

def create_artist_embeddings(tracks_df, embedding_dim=32):
    """
    Create artist embeddings based on their co-occurrence in playlists
    Using Word2Vec analogy: playlist = sentence, artist = word
    """
    from gensim.models import Word2Vec
    
    # Convert playlist URIs from string to list
    tracks_df['playlist_uris'] = tracks_df['playlist_uris'].apply(ast.literal_eval)
    
    # Create "sentences" where each sentence is a list of artists in a playlist
    playlist_artists = {}
    for _, row in tracks_df.iterrows():
        artist_uris = ast.literal_eval(row['artists_uris'])
        for playlist_uri in row['playlist_uris']:
            if playlist_uri not in playlist_artists:
                playlist_artists[playlist_uri] = []
            playlist_artists[playlist_uri].extend(artist_uris)
    
    # Train Word2Vec model
    artist_sequences = list(playlist_artists.values())
    model = Word2Vec(sentences=artist_sequences, 
                    vector_size=embedding_dim,
                    window=5,
                    min_count=1,
                    workers=4)
    
    return model

def prepare_track_feature_vector(track_row, all_unique_genres):
    """
    Creates a feature vector for a track combining:
    - Genre information (from artists)
    - Artist embedding (based on co-occurrence in playlists)
    """
    # Get genre vector
    genre_vector = vectorize_genre_list(track_row['artist_genres'], all_unique_genres)
    
    features = []
    
    # Add genre vector
    features.extend(genre_vector)
    
    return np.array(features)

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

def prepare_artist_data(filename):
    tracks = pd.read_csv(filename, nrows=10000)
    #print("Available columns:", tracks.columns.tolist())
    tracks = tracks.drop(columns=['Unnamed: 0', 'popularity', 'album_type', 'is_playable', 'release_date',  'playlist_uris'])
    tracks.head()

    #todo: for each artist url, query spotify api and get genre list. add to new column ['genres']

def get_track_features(sp, track_id):
    """
    Get audio features and basic track info from Spotify API
    """
    try:
        # Get audio features
        audio_features = sp.audio_features([track_id])[0]
        # Get track info including popularity
        track_info = sp.track(track_id)
        
        if audio_features:
            relevant_features = {
                'danceability': audio_features['danceability'],
                'energy': audio_features['energy'],
                'valence': audio_features['valence'],
                'tempo': audio_features['tempo'],
                'popularity': track_info['popularity'] / 100.0  # Normalize to 0-1 range
            }
            return relevant_features
    except:
        return None
    
def vectorize_track(track_genres, track_features, all_unique_genres):
    """
    Creates a feature vector combining genres and audio features
    """
    # Get genre vector (using your existing vectorize_genre_list function)
    genre_vector = vectorize_genre_list(track_genres, all_unique_genres)
    
    # Create feature vector from audio features
    feature_vector = np.array([
        track_features['danceability'],
        track_features['energy'],
        track_features['valence'],
        track_features['tempo'] / 200.0,  # Normalize tempo
        track_features['popularity']
    ])
    
    # Combine genre and feature vectors
    # You can adjust these weights based on importance
    genre_weight = 0.7
    features_weight = 0.3
    
    combined_vector = np.concatenate([
        genre_vector * genre_weight,
        feature_vector * features_weight
    ])
    
    return combined_vector

def recommend_tracks_ml(input_playlist_url, tracks_df, artist_df, n_recommendations=5):
    """
    Recommend tracks using a hybrid approach:
    1. Content-based: genres from input playlist
    2. Collaborative: artist embeddings
    """
    # Initialize Spotify client
    client_credentials_manager = SpotifyClientCredentials(
        client_id=os.environ.get('SPOTIFY_CLIENT_ID'),
        client_secret=os.environ.get('SPOTIFY_SECRET')
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    print("Starting recommendation process...")
    
    # Get genres from input playlist
    input_genres = get_genres_from_playlist(sp, input_playlist_url)
    print(f"Found {len(input_genres)} genres in input playlist")
    
    # Get all unique genres from artist_df
    all_unique_genres = extract_unique_genres(artist_df)
    print(f"Found {len(all_unique_genres)} unique genres in dataset")
    
    if not all_unique_genres:
        raise ValueError("No genres found in the dataset")
    
    # Create genre vector from input genres
    input_genre_vector = vectorize_genre_list(input_genres, all_unique_genres)
    
    # Create artist embeddings
    print("Creating artist embeddings...")
    artist_embedding_model = create_artist_embeddings(tracks_df)
    
    # Prepare feature vectors for all tracks
    print("Creating track vectors...")
    track_vectors = []
    track_ids = []
    
    for idx, track in tracks_df.iterrows():
        try:
            # Get track's artists' genres from artist_df
            artist_uris = ast.literal_eval(track['artists_uris'])
            track_genres = []
            
            # Collect genres from all artists of the track
            for artist_uri in artist_uris:
                artist_data = artist_df[artist_df['artist_uri'] == artist_uri]
                if not artist_data.empty:
                    artist_genres = artist_data['artist_genres'].iloc[0]
                    if isinstance(artist_genres, str):
                        artist_genres = ast.literal_eval(artist_genres)
                    track_genres.extend(artist_genres)
            
            if not track_genres:
                continue
            
            # Create basic feature vector from genres
            track_vector = vectorize_genre_list(track_genres, all_unique_genres)
            
            # Add artist embedding features
            artist_embeddings = []
            for artist_uri in artist_uris:
                if artist_uri in artist_embedding_model.wv:
                    artist_embeddings.append(artist_embedding_model.wv[artist_uri])
            
            if artist_embeddings:
                avg_artist_embedding = np.mean(artist_embeddings, axis=0)
                track_vector = np.concatenate([track_vector, avg_artist_embedding])
                
                track_vectors.append(track_vector)
                track_ids.append(track['track_uri'])
                
            if idx % 1000 == 0:
                print(f"Processed {idx} tracks...")
                
        except Exception as e:
            print(f"Error processing track {idx}: {str(e)}")
            continue
    
    if not track_vectors:
        raise ValueError("No valid track vectors could be created")
    
    print(f"Created vectors for {len(track_vectors)} tracks")
    
    # Convert to numpy array
    track_vectors = np.array(track_vectors)
    
    # Create input vector combining genre preferences and playlist context
    input_vector = np.zeros_like(track_vectors[0])
    
    # Get tracks from input playlist for context
    playlist_tracks = get_playlist_tracks(sp, input_playlist_url)
    
    # Average the vectors of tracks in the input playlist
    playlist_vectors = []
    for track in playlist_tracks:
        try:
            matching_tracks = tracks_df[tracks_df['track_uri'] == track['track_uri']]
            if matching_tracks.empty:
                continue
                
            track_data = matching_tracks.iloc[0]
            artist_uris = ast.literal_eval(track_data['artists_uris'])
            track_genres = []
            
            # Collect genres from all artists of the track
            for artist_uri in artist_uris:
                artist_data = artist_df[artist_df['artist_uri'] == artist_uri]
                if not artist_data.empty:
                    artist_genres = artist_data['artist_genres'].iloc[0]
                    if isinstance(artist_genres, str):
                        artist_genres = ast.literal_eval(artist_genres)
                    track_genres.extend(artist_genres)
            
            if not track_genres:
                continue
            
            track_vector = vectorize_genre_list(track_genres, all_unique_genres)
            
            # Add artist embedding
            artist_embeddings = []
            for artist_uri in artist_uris:
                if artist_uri in artist_embedding_model.wv:
                    artist_embeddings.append(artist_embedding_model.wv[artist_uri])
            
            if artist_embeddings:
                avg_artist_embedding = np.mean(artist_embeddings, axis=0)
                track_vector = np.concatenate([track_vector, avg_artist_embedding])
                playlist_vectors.append(track_vector)
                
        except Exception as e:
            print(f"Error processing playlist track: {str(e)}")
            continue
    
    # Combine genre preferences with playlist context
    if playlist_vectors:
        playlist_context_vector = np.mean(playlist_vectors, axis=0)
        genre_weight = 0.6
        context_weight = 0.4
        input_vector = (genre_weight * playlist_context_vector + 
                       context_weight * np.concatenate([input_genre_vector, 
                                                      np.zeros(len(playlist_context_vector) - len(input_genre_vector))]))
    else:
        input_vector = np.concatenate([input_genre_vector, 
                                     np.zeros(len(track_vectors[0]) - len(input_genre_vector))])
    
    # Use KNN with cosine similarity
    print("Finding nearest neighbors...")
    n_neighbors = min(len(track_vectors), max(n_recommendations * 2, 5))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(track_vectors)
    
    # Get recommendations
    distances, indices = knn.kneighbors([input_vector])
    
    recommended_tracks = []
    recommendation_distances = []
    
    for idx, distance in zip(indices[0], distances[0]):
        track_uri = track_ids[idx]
        recommended_tracks.append(track_uri)
        recommendation_distances.append(distance)
        if len(recommended_tracks) >= n_recommendations:
            break
    
    return {
        'recommended_tracks': recommended_tracks,
        'distances': recommendation_distances
    }

def get_track_names_from_uris(track_uris, sp):
    """
    Get track names and artists from URIs
    """
    track_info = []
    
    for i in range(0, len(track_uris), 50):
        batch = [uri.split(':')[-1] for uri in track_uris[i:i + 50]]
        tracks = sp.tracks(batch)['tracks']
        
        for track in tracks:
            track_info.append({
                'name': track['name'],
                'artists': ', '.join(artist['name'] for artist in track['artists'])
            })
    
    return track_info


def main():
    # Initialize Spotify client
    client_credentials_manager = SpotifyClientCredentials(
        client_id=os.environ.get('SPOTIFY_CLIENT_ID'),
        client_secret=os.environ.get('SPOTIFY_SECRET')
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Load datasets
    artist_df = pd.read_csv("artists.csv")
    tracks_df = pd.read_csv("final_tracks.csv")
    
    # Input playlist URL
    playlist_url = "https://open.spotify.com/playlist/2lvECTePT808JzzifupcbT?si=9de723f89db143a6"
    
    # User can choose between artist or track recommendations
    recommendation_type = "tracks"  # or "artists"
    
    try:
        if recommendation_type == "artists":
            # Get genre preferences from playlist
            genre_preferences = get_genres_from_playlist(sp, playlist_url)
            recommendations = recommend_artists(genre_preferences, artist_df)
            
            # Get artist names for display
            artist_names = get_artist_names_from_uris(sp, recommendations['recommended_artists'])
            
            print("\nRecommended artists:")
            for artist, distance in zip(artist_names, recommendations['distances']):
                print(f"Artist: {artist}, Distance: {distance:.3f}")
            
            print(f"\nTotal unique genres considered: {len(recommendations['all_genres'])}")
            
        else:  # tracks
            print("\nGenerating track recommendations based on playlist analysis...")
            recommendations = recommend_tracks_ml(playlist_url, tracks_df, artist_df)
            
            # Get track info for display
            track_info = get_track_names_from_uris(recommendations['recommended_tracks'], sp)
            
            print("\nRecommended tracks:")
            for info, distance in zip(track_info, recommendations['distances']):
                print(f"Track: {info['name']} by {info['artists']}, Distance: {distance:.3f}")
            
            print("\nRecommendation based on:")
            print("- Genre analysis")
            print("- Artist co-occurrence patterns")
            print("- Playlist context")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

if __name__ == "__main__":
    main()
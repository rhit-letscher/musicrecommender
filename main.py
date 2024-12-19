import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import ast
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import dotenv

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

def get_genres_from_playlist(playlist_url: str):
    """
    Retrieve all genres from a Spotify playlist through its artists.
    Returns a list of genres (non-unique to preserve frequency for weighting).
    
    Args:
        playlist_url (str): Full Spotify playlist URL or URI
        
    Returns:
        List[str]: List of all genres (including duplicates)
    """
    # Initialize Spotify client
    client_credentials_manager = SpotifyClientCredentials(
        client_id=os.environ.get('SPOTIFY_CLIENT_ID'),
        client_secret=os.environ.get('SPOTIFY_SECRET')
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Extract playlist ID from URL if needed
    if 'spotify.com' in playlist_url:
        playlist_id = playlist_url.split('/')[-1].split('?')[0]
    else:
        playlist_id = playlist_url
    
    # Get all tracks from playlist
    all_genres = []
    offset = 0
    batch_size = 100  # Spotify API limit for tracks per request
    
    while True:
        # Get batch of tracks
        results = sp.playlist_tracks(
            playlist_id,
            offset=offset,
            fields='items.track.artists,total',
            additional_types=['track']
        )
        
        if not results['items']:
            break
            
        # Extract artist IDs from the batch
        artist_ids = []
        for item in results['items']:
            if item['track'] and item['track']['artists']:
                artist_ids.extend([artist['id'] for artist in item['track']['artists']])
        
        # Get artist details in batches of 50 (Spotify API limit)
        for i in range(0, len(artist_ids), 50):
            artist_batch = artist_ids[i:i+50]
            artists = sp.artists(artist_batch)['artists']
            
            # Collect all genres from each artist
            for artist in artists:
                if artist['genres']:
                    all_genres.extend(artist['genres'])
        
        offset += batch_size
        
        # Check if we've processed all tracks
        if offset >= results['total']:
            break
    
    return all_genres

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

def prepare_artist_data(filename):
    tracks = pd.read_csv(filename, nrows=10000)
    #print("Available columns:", tracks.columns.tolist())
    tracks = tracks.drop(columns=['Unnamed: 0', 'popularity', 'album_type', 'is_playable', 'release_date',  'playlist_uris'])
    tracks.head()

    #todo: for each artist url, query spotify api and get genre list. add to new column ['genres']

def main():
    # Load the dataset
    
    df = pd.read_csv("artists.csv")
    prepare_artist_data("final_tracks.csv")
    # Create genre preferences
    playlist_url = "https://open.spotify.com/playlist/2lvECTePT808JzzifupcbT?si=9de723f89db143a6"
    genre_preferences = get_genres_from_playlist(playlist_url)
    #genre_preferences = ['pop', 'pop', 'rock', 'hyperpop_italiano', 'hyperpop_italiano', 'indie pop', 'indie pop', 'uk bass', 'uk bass']
    
    try:
        recommendations = recommend_artists(genre_preferences, df)
        
        #results
        recommendations['recommended_artists_names'] = get_artist_names_from_uris(recommendations['recommended_artists'])
        print("\nRecommended artists:")
        for artist, distance in zip(recommendations['recommended_artists_names'], 
                                  recommendations['distances']):
            
            print(f"Artist: {artist}, Distance: {distance:.3f}")

         #genre info   
        print(f"\nTotal unique genres: {len(recommendations['all_genres'])}")
        #print(f"All genres: {recommendations['all_genres']}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nDataset preview:")
        print(df.head())
        print("\nColumns:", df.columns.tolist())

if __name__ == "__main__":
    main()
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

def get_track_features(sp, track_id):
    """
    Get audio features and basic track info from Spotify API
    """
    try:
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

def get_artists_genres(sp: spotipy.Spotify, artists_df: pd.DataFrame):
    """
    Get artists genres from Spotify API
    """
    batch_size = 50
    all_genres = []
    
    # Process artists in batches
    for i in range(0, len(artists_df), batch_size):
        batch_artists = artists_df['artist_uri'].iloc[i:i + batch_size].tolist()
        # Extract artist IDs from URIs
        artist_ids = [uri.split(':')[-1] for uri in batch_artists]
        
        try:
            # Get artist details
            artists_data = sp.artists(artist_ids)['artists']
            
            # Extract genres for each artist
            batch_genres = [artist['genres'] if artist and 'genres' in artist else [] 
                          for artist in artists_data]
            
            all_genres.extend(batch_genres)
            
        except Exception as e:
            print(f"Error processing batch {i}-{i+batch_size}: {str(e)}")
            # Fill with empty lists for failed batch
            batch_genres = [[] for _ in range(len(batch_artists))]
            all_genres.extend(batch_genres)
            continue
        
        # Print progress
        if (i + batch_size) % 500 == 0:
            print(f"Processed {i + batch_size} artists...")
    
    # Add genres column to dataframe
    artists_df['genres'] = all_genres
        
    
def get_artist_names_from_uris(sp: spotipy.Spotify, artist_uris: list[str]):
    artist_names = []
    
    # Process URIs in batches of 50 (Spotify API limit)
    for i in range(0, len(artist_uris), 50):
        batch = artist_uris[i:i + 50]
        artists = sp.artists(batch)['artists']
        artist_names.extend([artist['name'] for artist in artists])
    
    return artist_names

def get_genres_from_playlist(sp: spotipy.Spotify, playlist_url: str):
    """
    Retrieve all genres from a Spotify playlist through its artists.
    Returns a list of genres (non-unique to preserve frequency for weighting).
    
    Args:
        playlist_url (str): Full Spotify playlist URL or URI
        sp (Spotipy.Spotify): authentified Spotify class instance to query the API with 
        
    Returns:
        List[str]: List of all genres (including duplicates)
    """
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


def get_playlist_tracks(sp: spotipy.Spotify, playlist_url: str):
    """
    Get all tracks from a playlist
    """
    
    if 'spotify.com' in playlist_url:
        playlist_id = playlist_url.split('/')[-1].split('?')[0]
    else:
        playlist_id = playlist_url
        
    tracks = []
    offset = 0
    
    while True:
        results = sp.playlist_tracks(playlist_id, offset=offset)
        if not results['items']:
            break
            
        for item in results['items']:
            if item['track']:
                tracks.append({
                    'track_uri': f"spotify:track:{item['track']['id']}",
                    'artist_uris': [f"spotify:artist:{artist['id']}" for artist in item['track']['artists']]
                })
                
        offset += len(results['items'])
        if offset >= results['total']:
            break
            
    return tracks
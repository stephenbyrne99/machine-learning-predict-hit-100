# this module will query the spotify api to return the spotify url of a song when provided with the key title and album
# https://developer.spotify.com/documentation/web-api/reference/search/search/ for documenetaion
import requests
from urllib.parse import quote as urlencode

# # required wrappers
# import spotipy 
# # Integrate the API with a spotify account
# from spotipy.oauth2 import SpotifyClientCredentials
# To read our credentials files
import json
import base64
import datetime


# set up our credentials
with open("authorisation.json", "r") as auth_file:
    auth_data = auth_file.read()

credentials = json.loads(auth_data)
client_id = credentials["client_id"]
client_secret = credentials["client_secret"]

# client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

# # create a spotify object to query their api
# sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


class SpotifyAPIWrapper(object):
    access_token = None
    token_expiry = datetime.datetime.now()
    client_id = None
    client_secret = None

    def __init__(self, client_id= credentials["client_id"], client_secret=credentials["client_secret"], *args, **kwargs):
        """
        If the client id and client secret are not provided then use the ones from our credentials file
        """
        super().__init__(*args, **kwargs)
        self.client_id = client_id 
        self.client_secret = client_secret


    def has_token_expired(self):
        """
        Returns true if a new spotify api token is required
        """
        if (self.token_expiry == None):
            print("Token has not been requested yet")
            return True
        
        now = datetime.datetime.now()
        if (now < self.token_expiry):
            return False
        else:
            return True



    def get_encoded_credentials(self):
        """
        Creates a base 64 encoded byte string to be used when making requests to the spotify api
        """    
        return base64.b64encode(f"{client_id}:{client_secret}".encode())


    def get_spotify_api_token(self):
        """
        Makes a request to the spotify api to get an access token, if an accesstoken is received, it is set and true returned,
        otherwise False is returned
        """
        token_url = "https://accounts.spotify.com/api/token"
        token_data = {
            "grant_type": "client_credentials"
        }
        # The Authorization takes the client_secret as a base64 encoded string.
        token_headers = {
            "Authorization": f"Basic {self.get_encoded_credentials().decode()}"
        }

        # make the request
        r = requests.post(token_url, data=token_data, headers=token_headers)
        if (r.status_code in range(200,299)):
            # set the token expiry
            response_data = r.json()
            now = datetime.datetime.now()
            self.access_token = response_data["access_token"]
            self.token_expiry = (now + datetime.timedelta(seconds=response_data["expires_in"]))
            return True
        else:
            return False


    def perform_token_check(self):
        """
        Checks if the client has a valid access token, if not it will fetch a new one
        """
        # check if our access token has expired
        if (self.has_token_expired()):
            if (not self.get_spotify_api_token()):
                print("Access token expired and we were unable to get a new one")
                return False
            else:
                print("New access token requested")
                return True
        else:
            return True

    def get_auth_headers(self):
        """
        Returns the required auth headers for an api request
        """
        return {
            "Authorization": f"Bearer {self.access_token}"
        }


    def keys_exist(self, element, *keys):
        """
        Check if *keys (nested) exists in "element" (dict)
        """
        if not isinstance(element, dict):
            raise AttributeError("Keys_exist() expects dict as first arguement")
        if len(keys) == 0:
            raise AttributeError("keys_exist() expects at least two arguments, one given")
            
        _element = element
        for key in keys:
            try:
                _element = _element[key]
            except (KeyError, IndexError) as e:
                return False
        return True

    def check_if_rate_limit_reached(self, response):
        """
        If a status code of 429 has been returned, then output in CAPS how long to wait to retry
        """
        if (response.status_code == 429):
            print("RATE LIMIT REACHED")
            print("Retry in " + response.headers["Retry-After"])

    def search_spotify_song_id(self, artist_name="", song_name=""):
        """
        Returns the spotify Uri for a given song name and artist name
        """
        # sanity check, you need to have the song_name
        if (song_name == ""):
            print("song_name not provided and is required for request")
            return False
        
        if (song_name == None):
            print("song name not rovided and is required")
            return False

        # check if our access token has expired
        if (not self.perform_token_check()):
            return False

        # construct query based on how much data has been provided
        search_endpoint = "https://api.spotify.com/v1/search?type=track&q="
        request_headers = {
            "Authorization": f"Bearer {self.access_token}"
        }

        # if a song name is provided then add it to the pile
        if (artist_name != ""):
            search_endpoint += urlencode(artist_name)
        
        if (song_name != ""):
            search_endpoint += "+" + urlencode(song_name)

        print("searching endpoint" + search_endpoint)
        
        try:
            r = requests.get(search_endpoint, headers=request_headers)
            if (r.status_code in range(200,299)):
                response_data = r.json()
                
                #check that the expected keys exist
                expected_keys = ["tracks", "items", 0, "id"]
                if (len(response_data["tracks"]["items"]) > 0 and self.keys_exist(response_data, *expected_keys)):
                    return response_data["tracks"]["items"][0]["id"]
                print("Song not found - " + artist_name + ", " + song_name)
                return False
            else:
                self.check_if_rate_limit_reached(r)
                return False
        except Exception as e:
            print("an exception occured during search" + e)
            return False


    def get_features_from_id(self, song_id):
        """
        TODO: implement a version of this function that accepts a list of tracks as an arguement
        if it fails then return none
        """
        # check token has expired logic
        if (not self.perform_token_check()):
            return False

        if (song_id == None or song_id == ""):
            print("Song id must be provided")
            return False

        features_endpoint = "https://api.spotify.com/v1/audio-features/" + song_id

        try:
            r = requests.get(features_endpoint, headers= self.get_auth_headers())
            if (r.status_code in range(200,299)):
                r_data = r.json()
                extracted_features = {
                    "id": r_data["id"],
                    "acousticness": r_data["acousticness"],
                    "danceability": r_data["danceability"],
                    "energy": r_data["energy"],
                    "instrumentalness": r_data["instrumentalness"],
                    "key": r_data["key"],
                    "liveness": r_data["liveness"],
                    "loudness": r_data["loudness"],
                    "mode": r_data["mode"],
                    "speechiness": r_data["speechiness"],
                    "tempo": r_data["tempo"],
                    "valence": r_data["valence"]
                }
                print(r_data)
                return extracted_features
            else:
                self.check_if_rate_limit_reached(r)
                return False
        except Exception as e:
            print(r.json())
            print(e)
            return False
    
    def get_song_ids_from_playlist(self, playlist_id):
        """
        Data will be returned in the form 
        [
            {
                song_name:
                artist_name:
                spotify_id:
            }
        ]
        """
         # check token has expired logic
        if (not self.perform_token_check()):
            return False

        if (playlist_id == None or playlist_id == ""):
            print("Song id must be provided")
            return False
        # get the tracks for this specific playlist id
        get_playlist_tracks_endpoint = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
        returned_tracks = []

        r = requests.get(get_playlist_tracks_endpoint, headers=self.get_auth_headers())
        if (r.status_code in range(200,299)):
            r_data = r.json()
            
            # For each of the tracks that are returned
            for track in r_data["tracks"]["items"]:
                track_info = {
                    "song_name": track["track"]["name"],
                    "artist_name": track["track"]["artists"][0]["name"],
                    "spotify_id": track["track"]["id"]
                }
                returned_tracks.append(track_info)
            
            print(returned_tracks)
            return returned_tracks
        else:
            return False


    
    def get_features_from_a_list_of_ids(self, list_of_ids):
        """
        Assume that the given list of ids is an array of strings, it must be for this algo to work.
        """
        # this will be appended to the url
        str_id_list = ",".join(list_of_ids)
        list_features_endpoint = "https://api/spotify.com/v1/audio-features/?ids=" + str_id_list
        
        if (not self.perform_token_check()):
            return None

        try:    
            r = requests.get(list_features_endpoint, headers = self.get_auth_headers())
            if (r.status_code in range(200,299)):
                r_data = r.json()
                
                return r_data["audio_features"]
            else:
                return None
        except Exception as e:
            print(e)
            return None





#### for testing our code
api = SpotifyAPIWrapper()
api.get_spotify_api_token()
# # api.get_features_from_id(api.search_spotify_song_id(song_name="Acid 444", artist_name="i_o"))
# # api.get_features_from_id(api.search_spotify_song_id(song_name="Feelin You", artist_name="DJ Spinn"))
print(api.get_features_from_id(api.search_spotify_song_id(song_name="Take a minute", artist_name="Billy Behan")))


# # UNICODE TESTS
# # The program crashes when it encounters unicode strings
# encoded_song = u"S\xc3\xa9bastien Roch".encode("latin1").decode("utf8")
# api.get_features_from_id(api.search_spotify_song_id(song_name="", artist_name=encoded_song))


# lOVE SONGS playlist link https://open.spotify.com/playlist/5KbTzqKBqxQRD8OBtJTZrS?si=uq0YSYzIQ3yzBB0y0BrKxQ
# api.get_song_ids_from_playlist("5KbTzqKBqxQRD8OBtJTZrS?si=uq0YSYzIQ3yzBB0y0BrKxQ")
"""
Calls genius API for lryics - web scraper

- authorisation must be added containing genius key

"""

from os import get_exec_path
import requests
import json
import datetime
import base64
import re
import os
from urllib.parse import quote as urlencode

from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

# get genius auth from our genius_auth.json file
with open("genius_auth.json", "r") as auth_file:
    auth_data = auth_file.read()

credentials = json.loads(auth_data)
client_id = credentials["client_id"]
client_secret = credentials["client_secret"]
token = credentials["token"]


class GeniusApi(object):
    access_token = None
    token_expiry = datetime.datetime.now()
    client_id = None
    client_secret = None

    user_agent = "cheethas@tcd.ie"

    def __init__(self, client_id= credentials["client_id"], client_secret=credentials["client_secret"], access_token=credentials["token"], *args, **kwargs):
        """
        If the client id and client secret are not provided then use the ones from our credentials file
        """
        super().__init__(*args, **kwargs)
        self.client_id = client_id 
        self.client_secret = client_secret
        self.access_token = access_token

    
    
    def get_access_token(self):
        # TODO: implement
        pass


    def get_auth_headers(self):
        """
        Returns the required auth headers for an api request
        """
        return {
            "Authorization": f"Bearer {self.access_token}"
        }


    def request_song_lyrics_url(self, song_name="", artist=""):
        """
        Leverages the Genius API to get the lyrics of a song
        """
        # sanity checks that the data provided is legit
        if (song_name == ""):
            print("song name not provided and it is required")
            return False
        
        if (song_name == None):
            print("song name not provided and it is required")
            return False
        
        search_endpoint = "https://api.genius.com/search"
        request_data = {
            "q": artist + song_name
        }
        try:
            r = requests.get(search_endpoint, data=request_data, headers=self.get_auth_headers())
            if (r.status_code in range(200,299)):
                r_data = r.json()
                # find a song with the artist name provided
                for match in r_data["response"]["hits"]:

                    if artist.lower() in match["result"]["primary_artist"]["name"].lower():
                        return match["result"]["url"] #id can be accessed in ["id"]
                
            return False
        except Exception as e:
            print(e)
            return False

    def get_lyrics_from_url(self, song_url):
        """
        Get the lyrics for a song from the song id
        """
        if (song_url == None):
            print("ERROR, passed id must not be like dis x")
            return False
        
        if (song_url == False):
            print("ERROR, song does not exist in genius services")
            return False
        
        try: 
            print(song_url)
            page = requests.get(song_url, headers={"User-Agent": "Mozilla/5.0"})
            # to trick genius into thinking we are a browser
            html = BeautifulSoup(page.content, "lxml")
        
            # lyrics_list = html.select('div[class="lyrics"]')
            
            # Genius returns random variants of its page to try and deter bots, here is a version that gets around this
            for tag in html.select('div[class^="Lyrics__Container"], .song_body-lyrics p'):

                for i in tag.select('i'):
                    i.unwrap()
                tag.smooth()

                t = tag.get_text(strip=True, separator='\n')
                if t:
                    return t


            # if (lyrics_list == None):
            #     return False 
            # if (lyrics_list == []):
            #     return False
            
            # lyrics = lyrics_list[0].get_text()

            # # if none, then try again with a different approach

            # # remove the paragraph headers
            # lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
            # # remove empty lines
            # lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])

            # # return clean lyrics
            # return lyrics
        except Exception as e:            
            print(e)
            return False


# # for testing the api wrapper made
api = GeniusApi()
# print(api.get_lyrics_from_url(api.request_song_lyrics_url(song_name="HISTORY", artist="Michael Jackson")))
# print(api.get_lyrics_from_url(api.request_song_lyrics_url(song_name="Feelin You", artist="DJ Spinn")))
# print(api.get_lyrics_from_url(api.request_song_lyrics_url(song_name="Atlas", artist="Bicep")))
# print(api.get_lyrics_from_url(api.request_song_lyrics_url(song_name="If I...", artist="Foxy Brown")))
# print(api.get_lyrics_from_url(api.request_song_lyrics_url(song_name="Bad Guy", artist="Billie Eilish")))

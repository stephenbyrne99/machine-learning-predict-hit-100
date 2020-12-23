import numpy as np
import pandas as pd
import h5py
import pathlib as Path

import hdf5_getters

def read_h5_file(file_path):
    with h5py.File(file_path) as hdf:
        ls = list(hdf.keys())
        print("List of datasets in this file", ls)

        musicbrainz = hdf.get("musicbrainz")
        print(musicbrainz.keys())

        songs = musicbrainz.get("songs")

        # data = hdf.get("metadata")
        # print("metadata", data.keys())
        # songs = data.get("songs")
        # for song in songs:
        #     print(song)

        

# The following code reads the artist names and song names from the dataset
# test on a random file form the milion song dataset
# file_string = "C:/Users/schee/Documents/University/4/Machine Learning/group assignment/dataset/millionsongsubset_full/MillionSongSubset/data/A/A/A/TRAAAAW128F429D538.h5"
# read_h5_file(file_string)
# data_path = Path("../dataset/millionsongsubset_full/MillionSongSubset/data/A/A/A/TRAAAAW128F429D538.h5")

# Read the whole database summary into one file
h5 = hdf5_getters.open_h5_file_read("C:/Users/schee/Documents/University/4/Machine Learning/group assignment/dataset/millionsongsubset_full/MillionSongSubset/AdditionalFiles/subset_msd_summary_file.h5")

number_of_rows = 1000

# preallocate the space for performance reasons
dataframe = pd.DataFrame(columns=("Artist name", "Title", "Year"), index=np.arange(0,number_of_rows ))
# create a csv file with the artist name and track title as headers
for k in range(1000):
    artist_name = hdf5_getters.get_artist_name(h5, k)
    song_name = hdf5_getters.get_title(h5, k)
    year = hdf5_getters.get_year(h5, k)
    
    # add the gotten data to the frame
    dataframe.loc[k] = [artist_name, song_name, year]


print(dataframe)


# what i want to do now, is take the first half of a song, and use spotify's search to return the first match for that song and artist, if it does not show up then no bodge.




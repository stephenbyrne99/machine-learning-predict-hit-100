# Scraping folder
This folder contains a collection of files which contributed to scraping information from the web, which was used to construct our datasets

## sourcing_data folder
This contains APIs that were written for the purpose of the project, to scrape information from the spotify api, genius api, and billboard api. Each of the apis are used in the notebooks below to fetch data related to each song.


### Files and their purposes 
**hdf5_getters**: This file is provided by the 1 million song dataset and is used in the project to extract the artist name and song name from our 1 millions song dataset sample   
**h5manipulation.py**: This file uses the hdf5_getters file above to create a csv file containing our artists and song names in the million song dataset subset. The reason for it's existance is that the million song dataset is stored in h5 format, this file completes the conversion.   
**give_data_tags**: Takes the artist name and song name from the million song dataset sample, and queries spotify's developer api, returning echo nest features for each of the songs. The spotify api is interacted with using our home-made api wrapper found in the sourcing_data folder  
**get_lyrics_for_songs**: Like the give_data_tags file, this file leverages genius' api to find if a song exists on their service, if it does, the webpage it exists on is requested, then scrapped for the lyrics. The returned lyrics are then cleaned.  
**give_readability_score**: Calulates the language of the returned lyrics, then calculates the gunning_fog index, and flesch readability scores for each of the songs lyrics.  
**clean_lyrics_dataset**: Removes all non-english songs from the dataset.

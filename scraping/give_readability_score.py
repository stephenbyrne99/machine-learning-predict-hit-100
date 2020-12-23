# Take the data from the dataset with lyrics and produce a readability score for the data
import numpy as np
import pandas as pd

# label each of the songs with text property values
from textstat import textstat

# to detect languages
from googletrans import Translator

# lang detect
from langdetect import detect

dataset = pd.read_csv("song_lyrics_withoutheaders_attempt_3_line_breaks_commas.csv")

lyrics = dataset["returned_lyrics"]
# loop through the dataset, add coloum for if it does not have lyrics
length = len(lyrics)
hasLyrics = []
language = []
flesch_readability = []
flesch_ease= []
gunning_fog = []

t = detect("hello world")
print(t)

# label if the dataset has lyrics or not
for i in range(0, length):
    if (lyrics[i] == ""):
        hasLyrics.append(0)
        language.append("None")
    else:
        hasLyrics.append(1)
        # get the language
        if (isinstance(lyrics[i], str)):
            try:
                t = detect(lyrics[i])
                language.append(t)
                if (t == "en"):
                    flesch_ease.append(textstat.flesch_reading_ease(lyrics[i]))
                    gunning_fog.append(textstat.gunning_fog(lyrics[i]))
                    flesch_readability.append(textstat.flesch_kincaid_grade(lyrics[i]))
                else:
                    flesch_ease.append("None")
                    gunning_fog.append("None")
                    flesch_readability.append("None")
            except Exception as e:
                language.append("None")
                flesch_ease.append("None")
                gunning_fog.append("None")
                flesch_readability.append("None")
        else:
            language.append("None")
            flesch_ease.append("None")
            gunning_fog.append("None")
            flesch_readability.append("None")


    
dataset["has_lyrics"] = hasLyrics
dataset["language"] = language
dataset["flesch_ease"] = flesch_ease
dataset["gunning_fog"] = gunning_fog
dataset["flesch_readability"] = flesch_readability

dataset.to_csv("lyrics_tags_v_5_dataset_novel", index=False)



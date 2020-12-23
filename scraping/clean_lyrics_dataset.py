
"""
Takes lyrics dataset and if no readability returns null replaces with 0's

"""

import pandas as pd

# takes in dataset with readability score
dataset = pd.read_csv("INSERT DATASET GATHERED WITH give_readability_score.py")

new_dataset = pd.DataFrame(columns=["Artist, Title, spotify_id, on_genius, genius_url, returned_lyrics, has_lyrics, language, flesch_ease, gunning_fog, flesch_readability"])

add_index = 0

new_dataset = dataset.loc[dataset['language'].isin(["en", "None"])]
    

print(new_dataset)

# now we have an english only dataset
# try and fix None labels
# we now want a new label, has lyrics
has_lyrics = []
for i, row in new_dataset.iterrows():
    flesch = row["flesch_ease"]
    if (flesch == "None"):
        has_lyrics.append(0)

        row["flesch_ease"] = 0
        row["gunning_fog"] = 0
        row["flesch_readability"] = 0

        new_dataset.loc[i] = row
    else:
        new_dataset.loc[i] = row
        has_lyrics.append(1)

new_dataset["has_lyrics"] = has_lyrics

print(new_dataset)
new_dataset.to_csv("outputfile.csv")
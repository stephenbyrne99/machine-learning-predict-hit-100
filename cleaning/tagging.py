import pandas as pd

# loop through million song subset (h5.out)
# if a song exists from this in the audiio hot 100 file
# tag it as 1 , if not tag it as 0

clean_df = pd.read_csv ('/lyrics_clean_fix_v_2.csv')
top100_df = pd.read_csv ('../scraping/key_datasets/all_billboard_hits.csv')

tags = []
for i, row in clean_df.iterrows():
    songid = str(row['Title']) + str(row['Artist name'])
    # check if song id exists in audio table if so - append 1 to array, if not append 0
    if songid in top100_df.songid.values:
        tags.append(1)
    else:
        tags.append(0)

clean_df['isTop100'] = tags
print(clean_df)

clean_df.to_csv('/lyrics_clean_fix_v_2_tagged.csv', index = False)
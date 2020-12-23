import numpy as np
import pandas as pd

# Removes bitestring b

df = pd.read_csv ('h5out.csv')
print(df)

if(df['Artist name'][0][0] == 'b'):
    df['Artist name'] = df['Artist name'].str[2:]
    df['Title'] = df['Title'].str[2:]

    df['Artist name'] = df['Artist name'].str[:-1]
    df['Title'] = df['Title'].str[:-1]

print(df)

df.to_csv('/h5out-cleaned.csv', index = False)
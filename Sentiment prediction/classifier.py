import pandas as pd
import json
import pickle

"""
Construct data frame:
"""

df_columns = [
    'text',
    'likes'
]
df = {column: [] for column in df_columns}


"""
Read json with responses:
"""

with open('./Spiders/ithappens.json') as f:
    for n, line in enumerate(f):
        
        record = json.loads(line)
        
        df['text'].append(record['text'])
        df['likes'].append(record['likes'])
        
        if n % 1000 == 0:
            print "Responses have been read:", n

df = pd.DataFrame(df).ix[:, df_columns]
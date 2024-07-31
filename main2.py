import re
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data/data.csv')
data.dropna(inplace=True)

cols_to_drop = ["id", "key", "mode", "release_date", "name", "artists"]
df = data.drop(columns=cols_to_drop, axis=1)

numerical_cols = ["year", "danceability", "duration_ms", "energy", "explicit", "instrumentalness", "liveness", "loudness", "popularity", "speechiness", "tempo", "valence"]
df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].min()) / (df[numerical_cols].max() - df[numerical_cols].min())
kmeans = KMeans(n_clusters=8, n_init="auto")

kmeans.fit(df)

df["cluster"] = kmeans.labels_
data["cluster"] = kmeans.labels_

#pca 

# pca = PCA(n_components=2)
# df_pca = pca.fit_transform(df.drop("cluster", axis=1))

# df["pca1"] = df_pca[:, 0]
# df["pca2"] = df_pca[:, 1]

# plt.scatter(df["pca1"], df["pca2"], c=df["cluster"], cmap="viridis")
# plt.xlabel("PCA1")
# plt.ylabel("PCA2")
# plt.title("KMeans Clustering")
# plt.show()

# recommender

def find_track_index(track_name):
    try:
        ind = data[data["name"] == track_name].index[0]
        return ind
    except IndexError:
        return None
    
def recommend(track_name):
    track_ind = find_track_index(track_name)
    
    track_cluster = data.loc[track_ind]["cluster"]

    filter = (data["cluster"] == track_cluster)
    filtered_df = data[filter]

    songs = []

    for i in range(5):
        recommendation = filtered_df.sample()

        songs.append(recommendation.iloc[0]["name"])

    return songs

print(recommend("Jailhouse Rock"))
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('data/cleaned_dataset.csv')
data.dropna(inplace=True)

cols_to_drop = ["Track", "Album", "Album_type", "Channel", "Licensed", "Title", "official_video", "most_playedon", "Views", "Artist", "Comments", "Likes"]
df = data.drop(columns=cols_to_drop)

numerical_cols = ["Loudness", "Tempo", "Duration_min", "Energy", "Instrumentalness", "Liveness", "EnergyLiveness", "Stream"]

df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].min()) / (df[numerical_cols].max() - df[numerical_cols].min())

print(df.head())

def optimise_kmeans(df, k_range):
    distortions = []
    means = []
    
    for k in range(1, k_range+1):
        kmeans = KMeans(n_clusters=k, n_init=k_range)

        kmeans.fit(df)

        distortions.append(kmeans.inertia_)
        means.append(k)
    
    #elbow plot

    plt.plot(means, distortions, 'o-')

    plt.xlabel('Clusters')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method showing the optimal k')
    plt.grid(True)
    plt.show()

optimise_kmeans(df, 10) 

kmeans = KMeans(n_clusters=3, n_init="auto")

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
        ind = data[data["Track"] == track_name].index[0]
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

        songs.append(recommendation.iloc[0]["Track"] + " by " + recommendation.iloc[0]["Artist"])

    return songs

print(recommend("Jailhouse Rock"))
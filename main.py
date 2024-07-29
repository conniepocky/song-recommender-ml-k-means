from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv('data/cleaned_dataset.csv')
data.dropna(inplace=True)

cols_to_drop = ["Track", "Album", "Album_type", "Channel", "Licensed", "Title", "official_video", "most_playedon", "Artist"]
df = data.drop(columns=cols_to_drop)

numerical_cols = ["Loudness", "Views", "Likes", "Comments", "Tempo", "Duration_min", "Energy", "Instrumentalness", "Liveness", "EnergyLiveness", "Stream"]

df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].min()) / (df[numerical_cols].max() - df[numerical_cols].min())

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

#optimise_kmeans(df, 10) 4 clusters

kmeans = KMeans(n_clusters=4, n_init="auto")

kmeans.fit(df)

df["cluster"] = kmeans.labels_
data["cluster"] = kmeans.labels_

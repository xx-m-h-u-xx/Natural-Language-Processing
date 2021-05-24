from jange import ops, stream, vis
ds = stream.from_csv(
    "https://raw.githubusercontent.com/jangedoo/jange/master/dataset/bbc.csv",
    columns="news",
    context_column="type",
)
# Extract clusters
result_collector = {}
clusters_ds = ds.apply(
    ops.text.clean.pos_filter("NOUN", keep_matching_tokens=True),
    ops.text.encode.tfidf(max_features=5000, name="tfidf"),
    ops.cluster.minibatch_kmeans(n_clusters=5),
    result_collector=result_collector,
)
# Get features extracted by tfidf and reduce the dimensions
features_ds = result_collector[clusters_ds.applied_ops.find_by_name("tfidf")]
reduced_features = features_ds.apply(ops.dim.pca(n_dim=2))

# Visualization
vis.cluster.visualize(reduced_features, clusters_ds)



#############################################################################

# import required libraries
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.datasets import load_files

# for reproducibility
random_state = 0 

DATA_DIR = "./bbc/"
data = load_files(DATA_DIR, encoding="utf-8", decode_error="replace", random_state=random_state)
df = pd.DataFrame(list(zip(data['data'], data['target'])), columns=['text', 'label'])

# Feature Extraction
vec = TfidfVectorizer(stop_words="english")
vec.fit(df.text.values)
features = vec.transform(df.text.values)


# Model Training
cls = MiniBatchKMeans(n_clusters=5, random_state=random_state)
cls.fit(features)

# predict cluster labels for new dataset
cls.predict(features)

# to get cluster labels for the dataset used while
# training the model (used for models that does not
# support prediction on new dataset).
cls.labels_


# Visualisation
# reduce the features to 2D
pca = PCA(n_components=2, random_state=random_state)
reduced_features = pca.fit_transform(features.toarray())

# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(cls.cluster_centers_)

plt.scatter(reduced_features[:,0], reduced_features[:,1], c=cls.predict(features))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')


# Evaluation with labelled dataset
from sklearn.metrics import homogeneity_score
homogeneity_score(df.label, cls.predict(features))

# Evaluation with unlabelled dataset 
from sklearn.metrics import silhouette_score
silhouette_score(features, labels=cls.predict(features))
df.head()

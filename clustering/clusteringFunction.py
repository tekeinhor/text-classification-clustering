from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn import metrics
from time import time


def get_cluster_kmeans(tfidf_matrix, num_clusters, labels):
    """
        returns list of kmeans clusters.

        Parameters:
            tfidf_matrix
            num_clusters : int
                Number of clusters.
        Returns: list of clusters
    """
    km = KMeans(n_clusters = num_clusters)
    sampleSize = len(labels)
    name = 'k-means'
    t0 = time()
    km.fit(tfidf_matrix)
    cluster_list = km.labels_.tolist()
    print('model\ttime\tinertia homo\tcompl\tv-meas\tARI\tAMI\tsilhouette \n\
        %s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), km.inertia_,
            metrics.homogeneity_score(labels, km.labels_),
            metrics.completeness_score(labels, km.labels_),
            metrics.v_measure_score(labels, km.labels_),
            metrics.adjusted_rand_score(labels, km.labels_),
            metrics.adjusted_mutual_info_score(labels,  km.labels_),
            metrics.silhouette_score(tfidf_matrix, km.labels_,metric='euclidean',sample_size=sampleSize)))
    print(80 * '_')
    return cluster_list


def multidim_scaling(similarity_matrix, n_components):
    """
        Returns the result of  matrix dimension Reduction.
        
        Parameters:
            similarity_matrix : matrix
            n_components : int
                Number of components.
        Returns: two-dimensional array.
    """
    one_min_sim = 1 - similarity_matrix
    mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=4)
    pos = mds.fit_transform(one_min_sim)  # shape (n_components, n_samples)
    x_pos, y_pos = pos[:, 0], pos[:, 1]
    return (x_pos, y_pos)


def pca_reduction(similarity_matrix, n_components = 10):
    """
    performs Principal component analysis  reduction.
    """
    one_min_sim = 1 - similarity_matrix
    pca = PCA(n_components)
    pos = pca.fit_transform(one_min_sim)
    x_pos, y_pos = pos[:, 0], pos[:, 1]
    return (x_pos, y_pos)


def tsne_reduction(similarity_matrix):
    """
    performs  t-distributed stochastic neighbor embedding reduction.
    """
    one_min_sim = 1 - similarity_matrix
    tsne = TSNE(learning_rate=1000).fit_transform(one_min_sim)
    x_pos, y_pos = tsne[:, 0], tsne[:, 1]
    return (x_pos, y_pos)
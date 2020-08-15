import os
import numpy as np
import pandas as pd

from enum import Enum
from scipy.spatial import distance
from scipy.sparse import csgraph, csr_matrix

from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from disjoint_set import DisjointSet

NMI_AVERAGE = 'geometric'


class DistanceSimilarity(Enum):
    """
    Enum for different identification of different similarity/distance measures that can be used for the adjacency matrix
    """
    MANHATTAN = 1
    EUCLIDEAN = 2
    GAUSSIAN = 3
    COSINE = 4


def delete_random_edge(A, row_sums, n):
    """Deletes a random edge in the adjacency matrix A

    creates a random value that corresponds to one edge in A and deletes it

    Parameters
    ----------
    A: numpy.array
        the adjacency matrix
    row_sums: numpy.array
        matrix of the row sums for a quicker identification which edge to delete
    n: int
        size of the dataset

    Returns
    -------
    target_row: int
        the row in A where the edge was deleted
    target_col: int
        the column in A where the edge was deleted
    value: float
        the weight of the edge

    """

    total_dist = row_sums[len(row_sums) - 1, 1]
    random_number = np.random.random() * total_dist

    target_row = np.argmax(row_sums[:, 1] > random_number)

    dist = row_sums[target_row, 1]
    target_col = 0
    for i in range(n - 1, -1, -1):
        if (dist < random_number):
            target_col = (i + 1)
            break
        dist -= A[target_row, i]

    value = A[target_row, target_col]
    A[target_row, target_col] = 0

    return target_row, target_col, value


def create_knn_adjacency(data, k, verbose=False):
    # NOTE! The Knn Adjacency Matrix is not symmetric
    if verbose:
        print("Creating knn Adjacency Matrix with k =", k)

    A = kneighbors_graph(data, n_neighbors=k, include_self=False, mode='distance')
    return A.toarray()


def create_specific_adjacency_matrix(data, distance_similarity=DistanceSimilarity.EUCLIDEAN, sigma=1, verbose=False):
    """
    A = Adjacency Matrix with euclidean distance, 0 on Diagonal Entries to not perturb distance sums
    """
    if verbose:
        print("Creating Adjacency with:", DistanceSimilarity(distance_similarity))

    n, dim = data.shape
    A = np.zeros(shape=(n, n), dtype=np.float32)

    for i in range(0, n):
        for j in range(i, n):

            if distance_similarity == DistanceSimilarity.MANHATTAN:
                dist = distance.cityblock(data[i, :], data[j, :])
            elif distance_similarity == DistanceSimilarity.GAUSSIAN:
                dist = np.exp(distance.euclidean(data[i, :], data[j, :]) / (2 * sigma ** 2)) - sigma
            elif distance_similarity == DistanceSimilarity.COSINE:
                dist = distance.cosine(data[i, :], data[j, :]) / 2
            else:
                dist = distance.euclidean(data[i, :], data[j, :])

            A[i, j] = dist

    return A


def create_row_sums(A, n):
    """Create a matrix with row sums for edge deletion

    Creates a (2,n) matrix that holds the edge weight sums of the adjacency matrix per row, this matrix is used later
    to identify which random edge is deleted

    Parameters
    ----------
    A: numpy.array
        the adjacency matrix, has to be the upper triangular matrix
    n:
        the size of the dataset

    Returns
    -------
        : numpy.array
        A numpy array holding cumulated and single sums of the adjacency matrix

    """
    rowSum = np.zeros(shape=(n, 2), dtype=np.float64)

    rowSum[:, 0] = np.sum(A, axis=1)

    cumm = 0
    for i in range(0, n):
        cumm += rowSum[i, 0]
        rowSum[i, 1] = cumm

    return rowSum


def load_csv_dataset(data_path, name, verbose=False):
    """helper function to make dataloading more convenient

    Parameters
    ----------
    data_path: str
        path to the data folder
    name: str
        name of the dataset to load from the data_path

    Returns
    -------
    : tuple
        tuple consisting of the dataset and the labels

    """
    abs_path = os.path.join(data_path, name)
    if verbose:
        print(abs_path)
    import_data = pd.read_csv(abs_path, header=None)

    rows, cols = import_data.shape

    labels = import_data.iloc[:, cols - 1]
    data = import_data.iloc[:, 0:cols - 1]

    if data.dtypes[0] == 'object':
        print(' >> string', name, data.dtypes[0])
        data = data.drop(data.columns[0], axis=1)

    data_loaded = (np.array(data), np.array(labels))

    return data_loaded


def random_link(dataset,
                noise=3,
                knn=0,
                distance_similarity=DistanceSimilarity.EUCLIDEAN,
                verbose=False,
                stopping_k=0):

    n = dataset.shape[0]

    max_stopping_score = 0
    deleted_edges_stack = []

    # Either create initial kNN Graph or a fully connected Graph
    if knn and knn > 0:
        distance_matrix = create_knn_adjacency(data=dataset, k=knn)

        start_components, _ = csgraph.connected_components(
            csr_matrix(distance_matrix + distance_matrix.transpose()),
            directed=False,
            return_labels=True)

    else:
        distance_matrix = create_specific_adjacency_matrix(data=dataset,
                                                                    distance_similarity=distance_similarity)
        start_components = 1

    a_mean = distance_matrix.mean()
    row_sums = create_row_sums(distance_matrix, n)

    if verbose:
        print(start_components, 'Connected Components at the beginning')

    ##################################################################
    # Delete Edges to define order
    cur_edges = 0
    while row_sums[n - 1, 1] > a_mean * 20:  # Delete Edges

        d_row, d_col, d_val = delete_random_edge(distance_matrix, row_sums, n)

        cur_edges += 1

        row_sums[d_row][0] -= d_val
        deleted_edges_stack.append((d_row, d_col))

        cumm = 0
        if d_row > 0:
            cumm = row_sums[d_row - 1, 1]

        for i in range(d_row, n):
            cumm += row_sums[i, 0]
            row_sums[i, 1] = cumm

    # Now we have deleted most of the edges and we want to add them in reverse order!
    n_components, n_labels = csgraph.connected_components(
        csr_matrix(distance_matrix + distance_matrix.transpose()),
        directed=False,
        return_labels=True)

    n_components = n

    ds = DisjointSet(n)
    ds.init_with_cluster(n=n, labels=list(n_labels))

    last_n_components = n_components

    comparison_clustering_results = dict()
    edge_index = len(deleted_edges_stack) - 1

    edges_added = 0

    while n_components > start_components:  # Add edges until connected components = 1

        if verbose:
            print(n_components, 'components')

        ds.add_edge(edge=deleted_edges_stack[edge_index])
        edge_index -= 1
        edges_added += 1

        if verbose:
            print(len(deleted_edges_stack), 'STACK SIZE')

        n_components = ds.get_components()

        if n_components != last_n_components:
            last_n_components = n_components

            n_labels = ds.get_labels()

            ################################################################
            # NOISE, Count labels that occur more than once to estimate how many non noise points we have
            if noise > 0:
                _, counts = np.unique(n_labels, return_counts=True)
                n_components_no_noise = len(counts[counts > noise])

                if n_components_no_noise < 1:
                    continue
            else:
                n_components_no_noise = n_components

            ##################################################################
            # STOPPING AT K COMPONENTS
            if stopping_k:
                if n_components_no_noise <= stopping_k:
                    best_labels_with_vote = n_labels
                    break

            ##################################################################
            # STOPPING CRITERION
            else:
                if n_components_no_noise not in comparison_clustering_results.keys():

                    pred = KMeans(n_clusters=n_components_no_noise, n_init=3, max_iter=30).fit(dataset)

                    comparison_clustering_results[n_components_no_noise] = pred

                else:
                    pred = comparison_clustering_results[n_components_no_noise]

                if n_components_no_noise > 1:

                    if verbose:
                        print(n_components_no_noise, 'No Noise')

                    stopping_score = normalized_mutual_info_score(n_labels, pred.labels_, average_method=NMI_AVERAGE)

                    if stopping_score >= max_stopping_score:
                        max_stopping_score = stopping_score
                        best_labels_with_vote = n_labels

    return best_labels_with_vote


if __name__ == '__main__':
    data, labels = load_csv_dataset('data', 'arrhythmia.txt')

    r_labels = random_link(data, distance_similarity=DistanceSimilarity.EUCLIDEAN, verbose=False)

    nmi = normalized_mutual_info_score(r_labels, labels, average_method=NMI_AVERAGE)

    print("NMI: {:.4f}".format(nmi))

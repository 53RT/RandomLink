import numpy as np


class DisjointSet:
    """Adapted Disjoint Set Data Structure

    adaptions are made so that the current number of connected components and their labelling is available after every
    operation at no extra cost.
    """

    def __init__(self, n):
        """Initializes the data structure for n objects

        (basically the make-set operation)

        Args:
            n: int
            the number of objects
        """
        self.n = n
        self.clusters = [{x} for x in list(range(0, n, 1))]
        # cluster indices as dict, per item
        self.cluster_indices = dict()
        self.n_components = n

        for idx in list(range(0, n, 1)):
            self.cluster_indices[idx] = [idx]

        list(range(0, n, 1))

    def __repr__(self):
        """returns state of the data structure"""

        return "Num Components: {}\nClusters: {}\nIndices: {}\n".format(self.n_components,
                                                                        self.clusters,
                                                                        self.cluster_indices)

    def init_with_cluster(self, n, labels):
        """Initializes the data structure with additional labels so that some objects can be already united.

        Initializes a new data structure and adds edges that lead to a labelling as provided

        Args:
            n: int
            number of objects overall
            labels: np.array
            numpy array with the shape (n,) providing class labels for each object

        """
        np_labels = np.array(labels)
        self.__init__(n)

        # Add Edges to create same labels
        for label in set(labels):
            to_cluster = np.where(np_labels == label)[0]

            for e in to_cluster[1:]:
                self.add_edge((to_cluster[0], e))

        for c in self.clusters:
            if c is not None:
                indices = sorted(list(c))
                for e in c:
                    self.cluster_indices[e] = indices

    def add_edge(self, edge):
        """unites two vertices with an edge

        basically the union operation by adding one edge

        Args:
            edge: tuple
            tuple containing indices to the two vertices

        Returns:

        """

        idx_c_1 = self.cluster_indices[edge[0]]
        idx_c_2 = self.cluster_indices[edge[1]]

        if idx_c_1 is not idx_c_2:

            if len(idx_c_2) > len(idx_c_1):
                idx_c_1, idx_c_2 = idx_c_2, idx_c_1

            self.clusters[idx_c_1[0]] = self.clusters[idx_c_1[0]].union(self.clusters[idx_c_2[0]])
            self.clusters[idx_c_2[0]] = None

            idx_c_1.extend(idx_c_2)

            for c_idx in idx_c_2:
                self.cluster_indices[c_idx] = idx_c_1

            self.n_components -= 1

    def get_components(self):
        return self.n_components

    def get_labels(self):
        return [v[0] for _, v in self.cluster_indices.items()]
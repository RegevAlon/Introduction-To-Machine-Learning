from collections import defaultdict
import pprint
import numpy as np
import kmeans

def singlelinkage(X, k):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    m, d = X.shape
    clusters = np.arange(m)
    while len(np.unique(clusters)) > k:
        print(len(np.unique(clusters)))
        min_distance = np.inf
        min_i, min_j = None, None
        for i in range(m):
            for j in range(i+1, m):
                if clusters[i] != clusters[j]:
                    distance = np.linalg.norm(X[i] - X[j])
                    if distance < min_distance:
                        min_distance = distance
                        min_i, min_j = i, j
        new_cluster = min(clusters[min_i], clusters[min_j])
        clusters[clusters == clusters[min_i]] = new_cluster
        clusters[clusters == clusters[min_j]] = new_cluster
    C = clusters
    C_labels = np.unique(C)
    for i,label in enumerate(C_labels):
        C[C==label] = i
    return C


def single_linkage_mnist(k, sample_size):
    mnist = np.load('mnist_all.npz')
    X = np.concatenate([mnist['train' + str(i)] for i in range(k)], axis=0)
    y = np.concatenate([i*np.ones(mnist['train' + str(i)].shape[0], dtype=int) for i in range(k)])

    sample_indices = np.random.choice(np.arange(X.shape[0]), sample_size, replace=False)
    X_sample = X[sample_indices, :]
    y_sample = y[sample_indices]
    C = singlelinkage(X_sample, k)

    sizes = np.bincount(C)

    labels = [np.bincount(y_sample[C == i]) for i in range(k)]

    stats = defaultdict(dict)

    for i in range(k):
        stats[i]['size'] = sizes[i]
        stats[i]['label'] = np.argmax(labels[i])
        stats[i]['percentage'] = labels[i][np.argmax(labels[i])] / sizes[i]

    error = 0
    for i in range(k):
        most_common_label = stats[i]['label']
        error += np.sum(y_sample[C == i] != most_common_label)
    error_rate = error / len(C)

    pprint.pprint(stats)

    print(f"Classification error rate: {error_rate}")


def single_with_data(k,x,y):
    X_sample = x
    y_sample = y
    C = singlelinkage(X_sample, k)

    sizes = np.bincount(C)

    labels = [np.bincount(y_sample[C == i]) for i in range(k)]

    stats = defaultdict(dict)

    for i in range(k):
        stats[i]['size'] = sizes[i]
        stats[i]['label'] = np.argmax(labels[i])
        stats[i]['percentage'] = labels[i][np.argmax(labels[i])] / sizes[i]

    error = 0
    for i in range(k):
        most_common_label = stats[i]['label']
        error += np.sum(y_sample[C == i] != most_common_label)
    error_rate = error / len(C)

    pprint.pprint(stats)

    print(f"Classification error rate: {error_rate}")


def single_kmeans_mnist(k, sample_size, t):
    x, y = devide_data(sample_size,k)
    kmeans.kmeans_with_data(k,x,y,t)
    single_with_data(k,x,y)




def devide_data(sample_size,k):
    mnist = np.load('mnist_all.npz')
    # flatten and concatenate all images into a single data matrix
    X = np.concatenate([mnist['train' + str(i)] for i in range(k)], axis=0)
    y = np.concatenate([i*np.ones(mnist['train' + str(i)].shape[0], dtype=int) for i in range(k)])
    # randomly select a sample of points
    sample_indices = np.random.choice(np.arange(X.shape[0]), sample_size, replace=False)
    X_sample = X[sample_indices, :]
    y_sample = y[sample_indices]
    return X_sample , y_sample


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    X = np.concatenate((data['train0'], data['train1']))
    m, d = X.shape

    # run singlelinkage
    c = singlelinkage(X, k=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    #simple_test()
    #single_linkage_mnist(10,300)
    #single_linkage_mnist(6,300)
    single_kmeans_mnist(6,300,10)


    # here you may add any code that uses the above functions to solve question 2

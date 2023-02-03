from collections import defaultdict
import random
import numpy as np
import pprint


random.seed(0)
def kmeans(X, k, t):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :param t: the number of iterations to run
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    m, d = X.shape
    centroids = X[np.random.choice(np.arange(m), k, replace=False), :]
    for _ in range(t):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        clusters = np.argmin(distances, axis=0)
        for i in range(k):
            centroids[i, :] = X[clusters == i, :].mean(axis=0)

    clusters = clusters.reshape(clusters.shape[0],1)
    return clusters


def kmeans_mnist(k, sample_size):
    mnist = np.load('mnist_all.npz')
    X = np.concatenate([mnist['train' + str(i)] for i in range(10)], axis=0)
    y = np.concatenate([i*np.ones(mnist['train' + str(i)].shape[0], dtype=int) for i in range(10)])

    sample_indices = np.random.choice(np.arange(X.shape[0]), sample_size, replace=False)
    X_sample = X[sample_indices, :]
    y_sample = y[sample_indices]
    t = 10
    C = kmeans(X_sample, k, t)
    C = C.reshape(C.shape[0])
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

def kmeans_with_data(k,x,y,t):
    X_sample = x
    y_sample = y
    C = kmeans(X_sample, k, t)
    C = C.reshape(C.shape[0])
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


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    X = np.concatenate((data['train0'], data['train1']))
    m, d = X.shape

    # run K-means
    c = kmeans(X, k=10, t=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"

if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    #simple_test()
    kmeans_mnist(10,1000)
    kmeans_mnist(6, 1000)

    # here you may add any code that uses the above functions to solve question 2

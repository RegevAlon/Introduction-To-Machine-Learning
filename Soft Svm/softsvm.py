import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
import cvxopt


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def softsvm(l, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    m, d = trainX.shape
    H = np.zeros((d + m, d + m))
    H[:d, :d] = np.eye(d) * 2 * l
    u = np.zeros(d + m)
    u[d:] = np.array([1 / m] * m)
    v = np.zeros(2 * m)
    v[m:] = np.ones(m)
    A = np.zeros((2 * m, m + d))
    A[:m, d:] = np.eye(m)
    A[m:, d:] = np.eye(m)
    A[m:, :d] = trainy[:, np.newaxis] * trainX
    solvers.options['show_progress'] = False
    sol = solvers.qp(matrix(H), matrix(u), matrix(-A), matrix(-v))
    return np.array(sol['x'][:d])


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    #i = np.random.randint(0, testX.shape[0])
    #predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    #print(f"The {i}'th test sample was classified as {predicty}")
    for i in range(len(testX)):
        print(float(np.sign(testX[i] @ w)))


def small_sample_experiment():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]
    final_results_test = []
    final_results_train = []

    for i in range(10):
        test_results = []
        train_results = []
        for j in range(10):
            error = 0
            # Get a random m training examples from the training set
            indices = np.random.permutation(trainX.shape[0])
            _trainX = trainX[indices[:m]]
            _trainy = trainy[indices[:m]]

            # run the softsvm algorithm

            w = softsvm(pow(10,i+1), _trainX, _trainy)

            for z in range(len(testX)):
                if np.sign(testX[z] @ w) != testy[z]:
                    error += 1
            test_results.append(error/len(testX))

            error = 0
            for z in range(len(_trainX)):
                if np.sign(_trainX[z] @ w) != _trainy[z]:
                    error += 1
            train_results.append(error / _trainX.shape[0])

        maxi = np.max(test_results)
        aver = np.average(test_results)
        mini = np.min(test_results)
        maxi_train = np.max(train_results)
        aver_train = np.average(train_results)
        min_train = np.min(train_results)


        final_results_train.append((maxi_train, aver_train, min_train))
        final_results_test.append((maxi, aver, mini))



    labels = ["10^" + str(i+ 1) for i in range(10)]
    max_error = [final_results_test[i][0] for i in range(10)]
    average_error = [final_results_test[i][1] for i in range(10)]
    min_error = [final_results_test[i][2] for i in range(10)]
    max_train_error = [final_results_train[i][0] for i in range(10)]
    average_train_error = [final_results_train[i][1] for i in range(10)]
    min_train_error = [final_results_train[i][2] for i in range(10)]


    width = 0.20
    x = np.arange(len(labels))  # the label locations
    fig, ax = plt.subplots()

    plt.plot(x, average_error,marker ='o',color="black", label="average error")
    plt.plot(x , average_train_error ,marker='o',color="purple", label="train error")
    ax.bar(x-(2*width), max_error, width=0.20, label='max test error')
    ax.bar(x-width, min_error , width=0.20, label='min test error')
    ax.bar(x, max_train_error , width=0.20, label='max train error')
    ax.bar(x+width, min_train_error , width=0.20, label='min train error')


    ax.set_xticks(x, labels)
    ax.set_title('Error defined by different lambda parameter and sample size 100')


    ax.set_ylabel('Error')
    ax.set_xlabel('Lambda Parameter')
    ax.legend(["average test error", "average train error", "max test error" ,"min test error", "max train error", "min train error"])
    plt.show()


def big_sample_experiment():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 1000
    d = trainX.shape[1]
    final_results_test = []
    final_results_train = []

    for i in range(1,9,2):
        i = 8 if i == 7 else i
        test_results = []
        train_results = []

        for j in range(10):
            error = 0
            # Get a random m training examples from the training set
            indices = np.random.permutation(trainX.shape[0])
            _trainX = trainX[indices[:m]]
            _trainy = trainy[indices[:m]]

            # run the softsvm algorithm

            w = softsvm(pow(10,i), _trainX, _trainy)

            for z in range(len(testX)):
                if np.sign(testX[z] @ w) != testy[z]:
                    error += 1
            test_results.append(error/testX.shape[0])

            error = 0
            for z in range(len(_trainX)):
                if np.sign(_trainX[z] @ w) != _trainy[z]:
                    error += 1
            train_results.append(error / _trainX.shape[0])

        maxi = np.max(test_results)
        aver = np.average(test_results)
        mini = np.min(test_results)
        aver_train = np.average(train_results)

        final_results_test.append((maxi, aver, mini))
        final_results_train.append(aver_train)

    labels = ["10^" + str(i) for i in range(1,7,2)] + ['10^8']
    #max_error = [final_results_test[i][0] for i in range(4)]
    average_error = [final_results_test[i][1] for i in range(4)]
    #min_error = [final_results_test[i][2] for i in range(4)]
    average_train_error = [final_results_train[i] for i in range(4)]

    x = np.arange(len(labels))  # the label locations
    fig, ax = plt.subplots()

    plt.scatter(x, average_error,marker ='o',color="black", label="average error")
    plt.scatter(x , average_train_error ,marker='o',color="purple", label="train error")

    ax.set_xticks(x, labels)
    ax.set_title('Error defined by different lambda parameter and sample size 1000')


    ax.set_ylabel('Error')
    ax.set_xlabel('Lambda Parameter')
    ax.legend(["average test error", "average train error"])
    plt.show()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    small_sample_experiment()
    big_sample_experiment()


    # here you may add any code that uses the above functions to solve question 2

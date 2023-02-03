import numpy as np
import matplotlib.pyplot as plt
import cvxopt
import softsvm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    m = trainX.shape[0]
    u = np.zeros((2 * m, 1))
    u[m:] = 1 / m
    v = np.zeros((2 * m, 1))
    v[m:] = 1
    G = create_gram_matrix(trainX, trainX, k)
    eps = 1e-4
    H = cvxopt.spdiag([cvxopt.matrix(2 * l * G), cvxopt.matrix(np.zeros((m, m)))]) + cvxopt.spdiag([eps] * 2 * m)
    A = np.zeros((2 * m, 2 * m))
    A[m:, :m] = trainy.reshape(m, 1) * G
    A[:m, m:] = np.eye(m)
    A[m:, m:] = np.eye(m)
    A = cvxopt.sparse(cvxopt.matrix(A)) + cvxopt.spdiag([eps] * 2 * m)
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(H, cvxopt.matrix(u), -A, cvxopt.matrix(-v))
    return np.asarray(sol['x'][:m])

def kernel(x, y, k):
  return (np.dot(x, y) + 1) ** k


def create_gram_matrix(X, Y, k):
    n_samples_X, n_features = X.shape
    n_samples_Y, _ = Y.shape
    G = np.zeros((n_samples_X, n_samples_Y))

    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            G[i, j] = (np.dot(X[i], Y[j]) + 1) ** k

    return G


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']
    print(trainX.shape)

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)
    print(w.shape)
    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"

def scatter_points():
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']
    x_axis = trainX[:, 0]
    y_axis = trainX[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(x_axis[trainy == 1], y_axis[trainy == 1], c='green')
    ax.scatter(x_axis[trainy == -1], y_axis[trainy == -1], c='red')
    ax.set_title('Scatter Plot of Training Data')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.show()

def compute_labels(alphas, X_samples, y, k, X_val):
  m = X_samples.shape[0]
  n = X_val.shape[0]
  G = create_gram_matrix(X_samples, X_val, k)

  return np.sign(np.dot(G.T, alphas))

def polysoftsvm_cross_validate(X, y, l_values, k_values, svm):
    n_samples = X.shape[0]
    fold_size = n_samples // 5
    errors = np.zeros((len(l_values), len(k_values)))
    stats = []

    for i, l in enumerate(l_values):
        for j, k in enumerate(k_values):
            error_sum = 0
            for fold in range(5):
                start = fold * fold_size
                end = (fold + 1) * fold_size
                X_validate = X[start:end]
                y_validate = y[start:end]
                X_train = np.concatenate((X[:start], X[end:]))
                y_train = np.concatenate((y[:start], y[end:]))
                alphas = svm(l, k, X_train, y_train)
                y_pred = compute_labels(alphas,X_train,y_validate,k,X_validate)
                error = np.sum(y_pred != y_validate.reshape(y_validate.shape[0], 1)) / len(y_validate)
                error_sum += error
                print(f"Fold {fold + 1} done for lambda = {l} & degree = {k} has {error} error")

            stats.append((str("lambda = "+str(l)),str("k = " + str(k)),error_sum / 5))
            errors[i, j] = error_sum / 5

    min_error = np.min(errors)
    stats = np.vstack(np.asmatrix(stats))
    print(stats)
    best_l, best_k = np.unravel_index(np.argmin(errors), errors.shape)
    best_l = l_values[best_l]
    best_k = k_values[best_k]
    best = "best error: " + str(min_error) + " best lambda: " + str(best_l) + " best degree: " + str(best_k)
    print(best)

def linearsoftsvm_cross_validate(X, y, l_values, svm):
    n_samples = X.shape[0]
    fold_size = n_samples // 5
    errors = np.zeros(len(l_values))
    stats = []

    for i, l in enumerate(l_values):
        error_sum = 0
        for fold in range(5):
            start = fold * fold_size
            end = (fold + 1) * fold_size
            X_validate = X[start:end]
            y_validate = y[start:end]
            X_train = np.concatenate((X[:start], X[end:]))
            y_train = np.concatenate((y[:start], y[end:]))
            w = svm(l,X_train, y_train)
            error = np.sum(np.sign(np.dot(X_validate, w)) != y_validate.reshape(y_validate.shape[0], 1)) / len(y_validate)
            error_sum += error
            print(f"Fold {fold + 1} done for lambda = {l} has {error} error")

        stats.append((str("lambda = " + str(l)), error_sum / 5))
        errors[i] = error_sum / 5


    min_error = np.min(errors)
    stats = np.vstack(np.asmatrix(stats))
    print(stats)
    best_l = np.unravel_index(np.argmin(errors), errors.shape)
    best_l = l_values[best_l[0]]
    best = "best error: " + str(min_error) + " best lambda: " + str(best_l)
    print(best)

def predictor_2d_grid(trainX, trainy, l_values, k_values):
    x1 = np.linspace(-8, 8, 100)
    x2 = np.linspace(-8, 8, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.c_[X1.ravel(), X2.ravel()]

    for i, l in enumerate(l_values):
        for j, k in enumerate(k_values):
            alphas = softsvmpoly(l, k, trainX, trainy)
            y_pred = compute_labels(alphas, trainX, trainy, k, X).reshape(X1.shape)
            plt.imshow(y_pred, cmap=ListedColormap(['red','blue']), extent=[-8, 8, -8, 8])
            patches = [mpatches.Patch(color='red', label='-1'), mpatches.Patch(color='blue', label='1')]
            plt.title(f'Poly soft svm prediction on a 2D grid with lambda = {l}, k = {k}')
            plt.legend(handles=patches)
            plt.show()

if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    #simple_test()
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']
    testX = data['Xtest']
    testy = data['Ytest']
    scatter_points()
    polysoftsvm_cross_validate(trainX,trainy,[1,10,100],[2,5,8],softsvmpoly)
    linearsoftsvm_cross_validate(trainX,trainy,[1,10,100],softsvm.softsvm)
    predictor_2d_grid(trainX, trainy,[100],[3,5,8])




    # here you may add any code that uses the above functions to solve question 4

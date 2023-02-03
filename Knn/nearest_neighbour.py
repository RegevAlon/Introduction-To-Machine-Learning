import numpy as np
import scipy
import matplotlib.pyplot as plt


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you


def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    classi = {}
    for i in range(len(y_train)):
        classi[tuple(x_train[i])] = y_train[i]
    classi["k"] = k
    return classi


def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """

    sol = np.zeros(len(x_test))
    i = 0
    for point1 in x_test:
        distance = []
        for point2 in classifier:
            if point2 != "k":
                dis = np.linalg.norm(point1 - point2)
                distance.append((dis, point2))

        distance.sort(key=lambda tup: tup[0])
        dic = {}
        for j in range(classifier.get("k")):
            vi = distance[j][1]
            val = classifier.get(tuple(vi))
            if dic.get(val) != None:
                dic[val] += 1
            else:
                dic[val] = 1

        label = max(dic, key=dic.get)
        sol[i] = label
        i += 1
    sol = sol.reshape(len(x_test), 1)

    return sol


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], len(test0) + len(test1) + len(test2) +len(test3))

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")

    for j in range(len(preds)):
        print(f"The {j}'th test sample was classified as {preds[j]}")


def Q2_1():

    data = np.load('mnist_all.npz')

    train0 = data['train2']
    train1 = data['train3']
    train2 = data['train5']
    train3 = data['train6']

    test0 = data['test2']
    test1 = data['test3']
    test2 = data['test5']
    test3 = data['test6']

    final_results = []
    for i in range(10, 110, 20):
        results = []
        for j in range(10):
            x_train, y_train = gensmallm([train0, train1, train2, train3], [2, 3, 5, 6], i)
            x_test, y_test = gensmallm([test0, test1, test2, test3], [2, 3, 5, 6], 100)
            classifer = learnknn(1, x_train, y_train)
            preds = predictknn(classifer, x_test)
            y_test = y_test.reshape(len(y_test),1)
            results.append(np.mean(preds != y_test))
            #results.append(np.square(np.subtract(preds,y_test)).mean())
        maxi = np.max(results)
        aver = np.average(results)
        mini = np.min(results)

        final_results.append((maxi, aver, mini))

    #labels = ['10', '20', '30', '40', '50','60', '70', '80', '90', '100']
    #max_error = [final_results[i][0] for i in range(10)]
    #average_error = [final_results[i][1] for i in range(10)]
    #min_error = [final_results[i][2] for i in range(10)]

    labels = ['10', '30', '50', '70', '90']
    max_error = [final_results[i][0] for i in range(5)]
    average_error = [final_results[i][1] for i in range(5)]
    min_error = [final_results[i][2] for i in range(5)]

    x = np.arange(len(labels))  # the label locations
    width = 0.20

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, max_error, width, label='max error')
    rects2 = ax.bar(x+ width, average_error, width, label='average error')
    rects3 = ax.bar(x + 2*width, min_error, width, label='min error')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Error')
    ax.set_xlabel('Sample Size')
    ax.set_title('Average Error defined by different sample Sizes')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    plt.ylim(0, max(max_error) + max(max_error)*0.25)

    fig.tight_layout()

    plt.show()
    return


def Q2_5():

    data = np.load('mnist_all.npz')

    train0 = data['train2']
    train1 = data['train3']
    train2 = data['train5']
    train3 = data['train6']

    test0 = data['test2']
    test1 = data['test3']
    test2 = data['test5']
    test3 = data['test6']

    final_results = []
    for i in range(1, 12):
        results = []
        for j in range(10):
            x_train, y_train = gensmallm([train0, train1, train2, train3], [2, 3, 5, 6], 200)
            x_test, y_test = gensmallm([test0, test1, test2, test3], [2, 3, 5, 6], 100)
            classifer = learnknn(i, x_train, y_train)
            preds = predictknn(classifer, x_test)
            y_test = y_test.reshape(len(y_test),1)
            results.append(np.mean(preds != y_test))
            #results.append(np.square(np.subtract(preds,y_test)).mean())
        maxi = np.max(results)
        aver = np.average(results)
        mini = np.min(results)

        final_results.append((maxi, aver, mini))

    labels = [i for i in range(1,12)]
    max_error = [final_results[i][0] for i in range(11)]
    average_error = [final_results[i][1] for i in range(11)]
    min_error = [final_results[i][2] for i in range(11)]

    #labels = ['10', '30', '50', '70', '90']
    #max_error = [final_results[i][0] for i in range(5)]
    #average_error = [final_results[i][1] for i in range(5)]
    #min_error = [final_results[i][2] for i in range(5)]

    x = np.arange(len(labels))  # the label locations
    width = 0.20

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, max_error, width, label='max error')
    rects2 = ax.bar(x+ width, average_error, width, label='average error')
    rects3 = ax.bar(x + 2*width, min_error, width, label='min error')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Error')
    ax.set_xlabel('K')
    ax.set_title('Average Error defined by different K value')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    plt.ylim(0, max(max_error) + max(max_error)*0.25)

    fig.tight_layout()

    plt.show()
    return


def Q2_6():
    def shuf(label):
        labels = [2, 3, 5, 6]
        labels.remove(label)
        return np.random.choice(labels)


    data = np.load('mnist_all.npz')

    train0 = data['train2']
    train1 = data['train3']
    train2 = data['train5']
    train3 = data['train6']

    test0 = data['test2']
    test1 = data['test3']
    test2 = data['test5']
    test3 = data['test6']

    final_results = []
    for i in range(1, 12):
        results = []
        for j in range(10):
            x_train, y_train = gensmallm([train0, train1, train2, train3], [2, 3, 5, 6], 200)
            x = np.random.randint(199, size=30)
            for sample in x:
                origin_label = y_train[sample]
                y_train[sample] = shuf(origin_label)
            x_test, y_test = gensmallm([test0, test1, test2, test3], [2, 3, 5, 6], 100)
            classifer = learnknn(i, x_train, y_train)
            preds = predictknn(classifer, x_test)
            y_test = y_test.reshape(len(y_test),1)
            results.append(np.mean(preds != y_test))
            #results.append(np.square(np.subtract(preds,y_test)).mean())
        maxi = np.max(results)
        aver = np.average(results)
        mini = np.min(results)

        final_results.append((maxi, aver, mini))

    labels = [i for i in range(1,12)]
    max_error = [final_results[i][0] for i in range(11)]
    average_error = [final_results[i][1] for i in range(11)]
    min_error = [final_results[i][2] for i in range(11)]

    #labels = ['10', '30', '50', '70', '90']
    #max_error = [final_results[i][0] for i in range(5)]
    #average_error = [final_results[i][1] for i in range(5)]
    #min_error = [final_results[i][2] for i in range(5)]

    x = np.arange(len(labels))  # the label locations
    width = 0.20

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, max_error, width, label='max error')
    rects2 = ax.bar(x+ width, average_error, width, label='average error')
    rects3 = ax.bar(x + 2*width, min_error, width, label='min error')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Error')
    ax.set_xlabel('K')
    ax.set_title('Average Error defined by K value and 15% corrupted train data')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    plt.ylim(0, max(max_error) + max(max_error)*0.25)


    fig.tight_layout()

    plt.show()
    return


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    #simple_test()
    print(Q2_1())
    print(Q2_5())
    print(Q2_6())
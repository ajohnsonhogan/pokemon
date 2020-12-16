import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import seaborn as sns
import pandas as pd
from type import Type


# A = (X^T*X)^-1*X^T
def getA(X):
    Xtranspose = X.transpose()
    dot_product = Xtranspose.dot(X)
    inverse = np.linalg.pinv(dot_product)
    A = inverse.dot(Xtranspose)
    return A


# W = A*t where A = (X^T*X)^-1*X^T
def getW(A, t):
    return A.dot(t)


# calculates the predicted type for each input row
def test(input, X, T):
    A = getA(X)
    result_labels = np.ones([len(input)])

    for i in range(len(input)):
        curr = input[i]
        curr_stats = np.asarray(curr[3:16]).astype(np.double)
        curr_stats = np.append(curr_stats, np.asarray(curr[17:18]).astype(np.double))
        curr_stats = np.append(curr_stats, 1)
        output_weights = np.zeros([18])
        # check probability of each type
        for j in range(18):
            currW = getW(A, T[j])
            output_weights[j] = curr_stats.dot(currW)
        # choose the max probability
        max_element = np.amax(output_weights)
        index = np.where(output_weights == max_element)
        result_labels[i] = index[0][0]
    return result_labels


# compares actual type with computed expected type
# along the diagonal is a match
def confusion_mat(original, result):
    cm = np.zeros([18, 18])
    for i in range(len(original)):
        cm[int(Type[original[i][1].upper()].value)][int(result[i])] += 1

    return cm.astype(int)


# sets the pyplot settings plots the dataframe as a seaborn heatmap
def show_heatmap(df, name):
    plt.figure(figsize=(10, 10))
    heatmap = sns.heatmap(df, annot=True, cmap="Blues")
    for i in range(18):
        heatmap.add_patch(matplotlib.patches.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=1))
    plt.ylabel("actual type")
    plt.xlabel("predicted type")
    plt.title(name)
    plt.show()


# shows the heatmaps for the absolute and normalized confusion matrices
# and prints the number correct out of the total
def show_confusion_matrix(conf, name):
    df = pd.DataFrame(data=conf, columns=[t.name for t in Type], index=[t.name for t in Type])
    normalized = (df.T / df.T.sum()).T

    show_heatmap(df, name)
    show_heatmap(normalized, name)

    num_correct = sum(conf[i, i] for i in range(18))
    total = sum(conf[i, j] for i in range(18) for j in range(18))
    print(name + ":", num_correct, "/", total, "correct")


def getT(pokemon):
    T = np.ones([18, len(pokemon) - 80])
    T *= -1

    for i in range(0, len(pokemon) - 80):
        p = pokemon[i + 1]
        T[Type[p[1].upper()].value, i] = 1
    return T


def getX(pokemon):
    stats = np.array(pokemon[1:723])
    stats = np.column_stack((stats[:, 3:16].astype(np.double), stats[:, 17:18].astype(np.double)))

    ones = np.ones((len(stats), 1))
    return np.column_stack((stats, ones))


def main():
    with open('pokemon.csv') as p:
        pokemon = list(csv.reader(p, delimiter=','))

    T = getT(pokemon)

    X = getX(pokemon)

    confusion = confusion_mat(pokemon[1:723], test(pokemon[1:723], X, T))

    show_confusion_matrix(confusion, "input dataset")

    with open('gen7.csv') as p:
        gen7 = list(csv.reader(p, delimiter=','))
    confusion = confusion_mat(gen7, test(gen7, X, T))

    show_confusion_matrix(confusion, "generation 7")


if __name__ == "__main__":
    main()

import csv
import numpy as np
from enum import Enum


class Type(Enum):
    BUG = 0
    DARK = 1
    DRAGON = 2
    ELECTRIC = 3
    FAIRY = 4
    FIRE = 5
    FIGHTING = 6
    FLYING = 7
    GROUND = 8
    GRASS = 9
    GHOST = 10
    ICE = 11
    NORMAL = 12
    POISON = 13
    WATER = 14
    PSYCHIC = 15
    ROCK = 16
    STEEL = 17


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
def test(input):
    A = getA(x)
    result_labels = np.ones([len(input)])

    for i in range(len(input)):
        curr = input[i]
        curr_stats = np.asarray(curr[3:10]).astype(np.double)
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


np.set_printoptions(linewidth=120)
np.set_printoptions(threshold=np.inf)
with open('pokemon.csv') as p:
    pokemon = list(csv.reader(p, delimiter=','))

T = np.ones([18, len(pokemon) - 1])
T *= -1

for i in range(0, len(pokemon) - 1):
    p = pokemon[i + 1]
    T[Type[p[1].upper()].value, i] = 1

    # if p[2] != '':
    #     T[Type[p[2].upper()].value, i] = 1

stats = np.array(pokemon[1:])
stats = stats[:, 3:10].astype(np.double)

#capture_rate,experience_growth,height_m,weight_kg
other = np.array(pokemon[1:])
other = other[:, 12:16].astype(np.double)

print(other)

ones = np.ones((len(stats), 1))
x = np.column_stack((stats, ones))

# with open('gen7.csv') as p:
#     gen7 = list(csv.reader(p, delimiter=','))
# confusion = confusion_mat(gen7, test(gen7))

confusion = confusion_mat(pokemon[1:], test(pokemon[1:]))
print(confusion)

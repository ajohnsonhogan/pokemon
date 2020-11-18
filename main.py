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


np.set_printoptions(linewidth=120)
np.set_printoptions(threshold=np.inf)
with open('pokemon.csv') as p:
    pokemon = list(csv.reader(p, delimiter=','))

T = np.ones([18, len(pokemon) - 1])
T *= -1

for i in range(0, len(pokemon) - 1):
    p = pokemon[i + 1]
    T[Type[p[1].upper()].value, i] = 1
    if p[2] != '':
        T[Type[p[2].upper()].value, i] = 1

stats = np.array(pokemon[1:])
stats = stats[:, 3:10].astype(np.double)

ones = np.ones((len(stats), 1))
x = np.column_stack((stats, ones))

print(x[:10])

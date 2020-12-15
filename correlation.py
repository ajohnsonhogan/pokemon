import pandas as pd
from pandas.plotting import parallel_coordinates
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from type import Type


with open('pokemon.csv') as p:
    pokemon = pd.read_csv(p)

pokemon["type1"] = pokemon.apply(lambda row: Type[row.type1.upper()], axis=1)
pokemon["type2"] = pokemon.apply(lambda row: Type[row.type2.upper()] if not pd.isna(row.type2) else None, axis=1)

for t in Type:
    pokemon["type1_" + t.name.lower()] = pokemon.apply(lambda row: 1 if row.type1 == t else 0, axis=1)

for t in Type:
    pokemon["type2_" + t.name.lower()] = pokemon.apply(lambda row: 1 if row.type2 == t else 0, axis=1)

plt.figure(figsize=(9, 9))
corr = pokemon.corr()
corr_lt = corr.where(np.triu(np.ones(corr.shape)).astype(np.bool).__invert__())
sns.heatmap(corr_lt, cmap="vlag", xticklabels=True, yticklabels=True)
plt.show()

# grouped stats line graph
pokemon["type1"] = pokemon.apply(lambda row: row.type1.name, axis=1)
grouped = pokemon.groupby(["type1"]).mean().reset_index()
plt.figure(figsize=(8, 4))
parallel_coordinates(grouped[
        ["type1", "attack", "defense", "hp", "sp_attack", "sp_defense", "speed"]
    ], "type1")
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.3, 1))
plt.show()

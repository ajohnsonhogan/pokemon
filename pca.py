import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from type import Type
import main
import csv

with open('pokemon.csv') as p:
    pokemon = pd.read_csv(p)

features = ["base_total", "attack", "defense", "hp", "sp_attack", "sp_defense", "speed", "base_egg_steps",
            "base_happiness", "capture_rate", "experience_growth", "height_m", "weight_kg", "generation", "is_legendary"
            ]

x = pokemon.loc[:, features].values
x = StandardScaler().fit_transform(x)
y = pokemon.loc[:, ["type1"]].values

pca = PCA()
pca.fit_transform(x)

# plots explained variance for each principal component
plt.plot(pca.explained_variance_)
plt.ylabel("explained variance")
plt.xlabel("principal component")
plt.show()

# with top 2 principal components...
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)

principalDf = pd.DataFrame(principal_components, columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, pokemon[['type1']]], axis=1)
finalDf["type1"] = finalDf.apply(lambda row: Type[row.type1.upper()].value, axis=1)


# scatter plot of each pokemon according to type based on the top 2 principal components
finalDf.plot.scatter(x="principal component 1",
                     y="principal component 2",
                     c="type1",
                     colormap="tab20c",
                     colorbar=False)
plt.title("actual type")
plt.show()

with open('pokemon.csv') as p:
    pokemon = list(csv.reader(p, delimiter=','))

T = main.getT(pokemon)
X = main.getX(pokemon)
result = main.test(pokemon[1:], X, T)

# scatter plot of each pokemon according to *predicted* type based on the top 2 principal components
finalDf.plot.scatter(x="principal component 1",
                     y="principal component 2",
                     c=result,
                     colormap="tab20c",
                     colorbar=False)
plt.title("predicted type")
plt.show()

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

allianz_filtre = pd.read_csv("allianz_filtre.csv")
df = allianz_filtre

features = ['char_size', 'height', 'width']

# Créer un objet StandardScaler
scaler = StandardScaler()

# Appliquer la normalisation aux données
df[features] = scaler.fit_transform(df[features])
x = allianz_filtre[['char_size', 'height', 'width']]

dbscan_model = DBSCAN(eps=0.5, min_samples=5)

print(x.columns)

labels = dbscan_model.fit_predict(x)

# Ajouter les étiquettes de cluster au DataFrame original
df['Cluster'] = labels

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Définir une colormap pour les clusters
cmap = plt.get_cmap('viridis')

# Utiliser les étiquettes de cluster pour attribuer une couleur à chaque point
sc = ax.scatter(df['char_size'], df['height'], df['width'], c=df['Cluster'], cmap=cmap)

# Ajouter une barre de couleur
plt.colorbar(sc, ax=ax, label='Cluster')

ax.set_title('DBSCAN Clustering (3D)')
ax.set_xlabel('char_size')
ax.set_ylabel('height')
ax.set_zlabel('width')
plt.show()
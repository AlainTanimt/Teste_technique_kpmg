import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

allianz_filtre = pd.read_csv("allianz_filtre.csv")
df = allianz_filtre

features = ['chars', 'height', 'width']

# Créer un objet StandardScaler
scaler = StandardScaler()

# Appliquer la normalisation aux données
df[features] = scaler.fit_transform(df[features])
x = df[features]
# print(x)

kmeans_model = KMeans(n_clusters=3, random_state=42)
labels = kmeans_model.fit_predict(x)

# Visualisation des clusters en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Utiliser les étiquettes de cluster pour attribuer une couleur à chaque point
sc = ax.scatter(df['char_size'], df['height'], df['width'], c=labels, cmap='viridis')

# Ajouter une barre de couleur
plt.colorbar(sc, ax=ax, label='Cluster')

ax.set_title('K-Means Clustering (3D)')
ax.set_xlabel('char_size')
ax.set_ylabel('height')
ax.set_zlabel('width')
plt.show()

centroids = kmeans_model.cluster_centers_

print(centroids[0])


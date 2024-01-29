from helper import produce_brut
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
def filtrer_dataframe(df):
    # Premier filtre: Supprimer les lignes où la valeur dans la colonne 'layout' est 'v'
    print(f"Avant le premier filtre : {len(df)} lignes")
    df = df[df['layout'] != 'v']
    print(f"Après le premier filtre : {len(df)} lignes")

    # Deuxième filtre: Enlever les redondances dans la colonne 'text'
    print(f"Avant le deuxième filtre : {len(df)} lignes")
    occurence_texte = df['text'].value_counts()
    df = df[df['text'].isin(occurence_texte[occurence_texte < 10].index)]
    print(f"Après le deuxième filtre : {len(df)} lignes")

    # Troisième filtre: Enlever les lignes contenant des chiffres dans la colonne 'text'
    print(f"Avant le troisième filtre : {len(df)} lignes")
    indices_a_supprimer = pd.to_numeric(df['text'], errors='coerce', downcast='integer').notna()
    df = df[~indices_a_supprimer]
    print(f"Après le troisième filtre : {len(df)} lignes")

    # Quatrième filtre: Conserver les lignes où la colonne 'chars' est supérieure à 5
    print(f"Avant le quatrième filtre : {len(df)} lignes")
    df = df[df['chars'] > 5]
    print(f"Après le quatrième filtre : {len(df)} lignes")

    return df

def k_means (df):
    features = ['chars', 'height', 'width']
    scaler = StandardScaler()

    # Appliquer la normalisation aux données
    df[features] = scaler.fit_transform(df[features])
    x = df[features]

    kmeans_model = KMeans(n_clusters=3)
    df['label'] = kmeans_model.fit_predict(x)
    centroids = kmeans_model.cluster_centers_
    ind = np.argsort((centroids[:,0]))
    cluster_labels = {ind[-1]: 'Paragraphe', ind[0]: 'Inutile', ind[1]: 'Titre',}
    df['label'] = df['label'].map(cluster_labels)

    df[features] = scaler.inverse_transform(df[features])
    return

def labeliseur (df):
    df_copie = df.copy()
    df_copie = filtrer_dataframe(df_copie)
    k_means(df_copie)
    df = pd.merge(df, df_copie[['Unnamed: 0', 'label']], on='Unnamed: 0', how='left')
    df['label'] = df['label'].fillna('Inutile')
    return df
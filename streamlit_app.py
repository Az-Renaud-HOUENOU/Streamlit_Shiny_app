import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

st.title("Exploration de la base de données IRIS")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].apply(lambda x: iris.target_names[x])

st.header("Aperçu des données")
st.write(df.head())

st.header("Statistiques descriptives")
st.write(df.describe())

st.header("Graphiques interactifs")
option = st.selectbox("Choisissez la variable à analyser :", df.columns[:-2])

st.subheader(f"Distribution de {option}")
fig, ax = plt.subplots()
sns.histplot(df[option], kde=True, ax=ax, color="blue")
st.pyplot(fig)

st.subheader("Pairplot des caractéristiques")
fig = sns.pairplot(df, hue="species", diag_kind="kde", palette="Set2")
st.pyplot(fig)

import streamlit as st
import pandas as pd
from datetime import datetime

st.title('Application de Lise LAROCHETTE')

st.write("Dataframe")

link = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/cars.csv"
df_voitures = pd.read_csv(link)

#afficher la DF
df_voitures


#créer un filtre

# Créer un bouton pour filtrer par continent "Europe."
if st.button("Filtrer par continent 'Europe.'"):
    df_filtered = df_voitures[df_voitures['continent'].str.contains('Europe')]
    # Afficher le DataFrame filtré
    st.write("DataFrame après le filtre :")
    st.write(df_filtered)

if st.button("Filtrer par continent 'US.'"):
    df_filtered = df_voitures[df_voitures['continent'].str.contains('US')]
    # Afficher le DataFrame filtré
    st.write("DataFrame après le filtre :")
    st.write(df_filtered)

if st.button("Filtrer par continent 'Japan.'"):
    df_filtered = df_voitures[df_voitures['continent'].str.contains('Japan')]
    # Afficher le DataFrame filtré
    st.write("DataFrame après le filtre :")
    st.write(df_filtered)


#afficher un graphique en bar
st.title("Puissance des voitures (cylindres) par nationalité")
st.bar_chart(data=df_voitures, x='continent', y='cylinders')
st.write("On remarque que les voitures américaines sont les plus puissantes.")

#afficher un graphique en courbe
import numpy as np

st.title("Consommation des voitures en fonction de la puissance (cylindres)")
st.line_chart(data=df_voitures, x= 'weightlbs', y='mpg')
st.write("On remarque que les voitures qui ont un poids entre 2000 et 2800 lbs consomment le plus.")




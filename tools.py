# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:48:44 2022

Nettoyer les données Openclassrooms 

@author: tomy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# ---------------------------------------------------------------------------
# -- DESCRIPTION DU JEU DE DONNEES
# ---------------------------------------------------------------------------

def get_info_data(df):
    print("------------------------------------------------------------------")
    print(f"LES DONNES \n")
    print("Nombre de lignes :", df.shape[0], "lignes")
    print("Nombre de colonnes :", df.shape[1], "colonnes")
    # Recherche de doublon
    print(
        f"Nombre d'éléments dupliqués dans le dataframe : {df.duplicated().sum()} eléments"
    )
    nb_na = df.isnull().sum().sum()
    if nb_na > 0:
        nb_data_tt = np.product(df.shape)
        pourcent_na_tt = round((nb_na / nb_data_tt) * 100, 2)
        print(
            f"Nombre total de données manquantes dans le dataframe : {nb_na} données manquantes sur {nb_data_tt} ({pourcent_na_tt}%)"
        )
    else:
        print("Aucune valeur manquante")

# ---------------------------------------------------------------------------

def get_description_variables(dataframe, type_var='all'):
    """
    Fonction qui retourne la description des variables qualitatives, quantitatives ou les deux
    Entrées
    Paramettres
    ----------
    @param IN :
        - dataframe
        - type_var = 'all'   : tous les types de variable
                     'categ' : type categorie / labelle
                     'num'   : type numérique
    Sortie : description des variables
    """
    if type_var == 'num':
        res = dataframe.describe()
    elif type_var == 'categ':
        res = dataframe.describe(exclude=[np.number])
    else:
        res = dataframe.describe(include='all')

    return res.T

# --------------------------------------------------------------------
# -- TYPES DES VARIABLES
# --------------------------------------------------------------------

def get_types_variables(dataframe, types, type_par_var, graph):
    """ Permet un aperçu du type des variables
    Parametres
    ----------
    @param IN : - dataframe
                - types : Si True lance dtypes, obligatoire
                - type_par_var : Si True affiche tableau des types de
                               chaque variable, obligatoire
                - graph : Si True affiche pieplot de répartition des types

    """

    if types:
        # 1. Type des variables
        print("-------------------------------------------------------------")
        print("Type de variable pour chacune des variables\n")
        display(dataframe.dtypes)

    if type_par_var:
        # 2. Compter les types de variables
        print("\n----------------------------------------------------------")
        print("Répartition des types de variable\n")
        values = dataframe.dtypes.value_counts()
        nb_tot = values.sum()
        percentage = round((100 * values / nb_tot), 2)
        table = pd.concat([values, percentage], axis=1)
        table.columns = [
            'Nombre par type de variable',
            '% des types de variable']
        display(table[table['Nombre par type de variable'] != 0]
                .sort_values('% des types de variable', ascending=False)
                .style.background_gradient('seismic'))

    if graph:
        # 3. Schéma des types de variable
        # print("\n----------------------------------------------------------")
        # Répartition des types de variables
        plt.pie(x = dataframe.dtypes.value_counts(),
        labels = dataframe.dtypes.value_counts().index,
        shadow=True,
        autopct='%1.1f%%',
        explode=[0,0.1])
        # plt.title('Répartion des types de variables',fontweight='bold')
        plt.show()

# ---------------------------------------------------------------------------
# -- VALEURS MANQUANTES
# ---------------------------------------------------------------------------

# On crée une fonction
def get_missing_value(df,afficher_pourcentage,afficher_heatmap):
    """ Permet un aperçu du type des variables
    Parametres
    ----------
    @param IN : df_work : dataframe, obligatoire
                types : Si True lance dtypes, obligatoire
                type_par_var : Si True affiche tableau des types de
                               chaque variable, obligatoire
                graph : Si True affiche pieplot de répartition des types

    """
    nb_na = df.isnull().sum().sum()
    nb_data_tt = np.product(df.shape)
    pourcent_na_tt = round((nb_na / nb_data_tt) * 100, 2)
    print(f"Nombre total de données manquantes dans le dataframe : {nb_na} données manquantes sur {nb_data_tt} ({pourcent_na_tt}%)")
    print("-------------------------------------------------------------")
    
    # Affichage du nombre et du pourcentage de valeurs manquantes par variable
    if afficher_pourcentage:
        print("Nombre et pourcentage de valeurs manquantes par variable\n")
        values = df.isnull().sum()
        percentage = 100 * values / len(df)
        table = pd.concat([values, percentage.round(2)], axis=1)
        table.columns = [
            'Nombres de valeurs manquantes',
            '% de valeurs manquantes']
        display(table[table['Nombres de valeurs manquantes'] != 0]
                .sort_values('% de valeurs manquantes', ascending=False)
                .style.background_gradient('YlOrRd'))
    print("------------------------------------------------------------------")
    # Vusualisation des données manquantes
    if afficher_heatmap:
        print("Visualisation des données manquantes")
        plt.figure(figsize=(20, 10))
        plt.title("Visualisation des valeurs manquantes")
        sns.heatmap(df.isna(), cbar=False)
        plt.show()
        
# ------------------------------------------------------------------------
def graph_NAN(data,seuil_na=20,debut=1,fin=50,title='titre'):
    """ Permet de mieux visualiser les % de NaN sur un nombre de colonne limité
    Parametres
    ----------
    @param IN : df : dataframe
                seuil : visualiser un ligne de seuil
                debut : indice de la colonne de début de l'intervalle
                fin   : indice de la dernière colonne a prendre en compte

    """
    plt.figure(figsize=(16,10))
    df=data.iloc[:,debut:fin]
    ax = plt.subplot(1,1,1)
    perc = (df.isnull().sum()/df.shape[0])*100
    
    ax = sns.barplot(x=df.columns,y=perc,color='steelBlue')
    if seuil_na != False:
        plt.axhline(y=seuil_na, color='r', linestyle='-')
    ax.set_title(title,fontsize=20, weight='bold')
    ax.set_xlabel('Colonnes',fontsize=20)
    ax.set_ylabel('% de NaN',fontsize=20)
    ax.set_xticklabels(df.columns,rotation=90)
    
plt.show()

# ------------------------------------------------------------------------
def get_null_factor(df, tx_threshold):
    """ Permet de choisir et visualiser les taux de NaN 
    Parametres
    ----------
        @param IN : df           : dataframe
                    tx-threshold : = seuil : visualiser un ligne de seuil

    """ 
    null_rate = ((df.isnull().sum() / df.shape[0])*100).sort_values(ascending=False).reset_index()
    null_rate.columns = ['Variables','Taux_de_Null']
    high_null_rate = null_rate[null_rate.Taux_de_Null >= tx_threshold]
    return high_null_rate

# ------------------------------------------------------------------------
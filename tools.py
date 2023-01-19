# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:48:44 2022

projet-3

@author: Philippe Moty
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from wordcloud import WordCloud
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import dendrogram

from scipy.stats import shapiro, normaltest, anderson
from scipy.stats import chi2_contingency
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# -- DESCRIPTION DU JEU DE DONNEES
# ---------------------------------------------------------------------------

def get_info_data(df):
    """
    Fonction qui affiche la taille du dataset
    """
    print("------------------------------------------------------------------")
    print("Taille du jeu de données \n")
    print("Nombre de lignes :", df.shape[0], "lignes")
    print("Nombre de colonnes :", df.shape[1], "colonnes")
    print("------------------------------------------------------------------")
    
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
                     'categ' : type categorie / label
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
        autopct='%1.1f%%'
        # explode=[0,0.1]
        )
        # plt.title('Répartion des types de variables',fontweight='bold')
        plt.show()

# ----------------------------------------------------------------------------


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
def graph_NAN_with_start_end(data,seuil_na=20,debut=1,fin=50,title='titre'):
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
        plt.text(len(df.isnull().sum()/len(df))/1.7, seuil_na+5, 
                 'Colonnes avec plus de %s%s missing values' %( seuil_na, '%'), 
                 fontsize=12,weight='bold', color='crimson', ha='left' ,va='top')

    ax.set_title(title,fontsize=20, weight='bold')
    ax.set_xlabel('Colonnes',fontsize=20)
    ax.set_ylabel('% de NaN',fontsize=20)
    ax.set_xticklabels(df.columns,rotation=90)
    
plt.show()
# ------------------------------------------------------------------------
def graph_NAN(df,seuil_na=20,title='titre'):
    """ Permet de mieux visualiser les % de NaN sur un nombre de colonne limité
    Parametres
    ----------
    @param IN : df : dataframe
                seuil : visualiser un ligne de seuil
                debut : indice de la colonne de début de l'intervalle
                fin   : indice de la dernière colonne a prendre en compte

    """
    plt.figure(figsize=(16,10))
    ax = plt.subplot(1,1,1)
    perc = (df.isnull().sum()/df.shape[0])*100
    
    ax = sns.barplot(x=df.columns,y=perc,color='steelBlue')
    if seuil_na != False:
        plt.axhline(y=seuil_na, color='r', linestyle='-')
        plt.text(len(df.isnull().sum()/len(df))/1.7, seuil_na+5, 
                 'Colonnes avec plus de %s%s missing values' %( seuil_na, '%'), 
                 fontsize=12,weight='bold', color='crimson', ha='left' ,va='top')

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
    #null_rate = ((df.isnull().sum() / df.shape[0])*100).sort_values(ascending=False).reset_index()
    null_rate = ((df.isnull().mean()*100)).sort_values(ascending=False).reset_index()
    null_rate.columns = ['Variables','Taux_de_Null']
    high_null_rate = null_rate[null_rate.Taux_de_Null >= tx_threshold]
    return high_null_rate

# ------------------------------------------------------------------------

def search_componant(df, suffix='_100g'):
    """ Permet d'isoler les données suffixé par _100g à mettre a zero si nan
    
    Parametre
    ----------
    df     : 
    suffix : 
        DESCRIPTION. The default is '_100g'.
    Sortie les élémentsuffixé par _100g

    """
    componant = [] 
    for col in df.columns:
         if '_100g' in col: componant.append(col)
         df_subset_columns = df[componant]
    return df_subset_columns

# ---------------------------------------------------------------------------
# -- INFORMATION SUR LES COLONNES
# ---------------------------------------------------------------------------
def get_info_colonne(df,colonne):
    '''
    Présente les informations sur la colonne : unique, nunique, na%
    parametres
    - df 
    - colonne 'ma_colonne'
    - nb_max : nombre qui permet de limiter l'affichage si trop élevé
    '''
    df_work = df[colonne]
    print(f"Colonne : {colonne}")
    print("________")
    print(f"Nombre de valeurs uniques : {df_work.nunique()}")
    print(f"NaN : {round(df_work.isnull().mean()*100,2)} %")
    print("--------")
    print("Les valeurs contenues dans la colonne")
    print(f"{df_work.unique()}")

# ---------------------------------------------------------------------------
def info_colonne_df_col_nb_max(df,colonne,nb_max):
    '''
    Présente les informations sur la colonne : unique, nunique, na%
    parametres
    - df 
    - colonne 'ma_colonne'
    - nb_max : nombre qui permet de limiter l'affichage si trop élevé
    '''
    df_work = df[colonne]
    print(f"Colonne : {colonne}")
    print("________")
    print(f"Nombre de valeurs uniques : {df_work.nunique()}")
    print(f"NaN : {round(df_work.isnull().mean()*100,2)} %")
    if df_work.nunique()<nb_max:
        print("--------")
        print("Les valeurs contenues dans la colonne")
        df_work.unique()
    elif df_work.nunique()>nb_max:
        print("Le nombre de valeur unique contenu dans cette colonne est supérieur au nb_max renseigné")

# ---------------------------------------------------------------------------
def info_list_colonnes(df,liste_col):
    '''
    '''
    for i in liste_col:
        
        df_work = df[i]
        print("________")
        print(f"Colonne : {i}")
        print(f"Nombre de valeurs uniques : {df_work.nunique()}")
        print(f"Type de la variable       : {df[i].dtypes}")
        print(f"NaN                       : {round(df_work.isnull().mean()*100,2)} %")
        print(f"Valeurs unique            : {df[i].unique()}")
        print("")
    
    

# ---------------------------------------------------------------------------

def stat_descriptives(dataframe, liste_variables):
    """
    Statistiques descriptives moyenne, mediane, variance, écart-type,
    skewness et kurtosis du dataframe transmis en paramètre
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                liste_variables : colonne dont on veut voir les stat descr
    @param OUT : dataframe des statistiques descriptives
    """
    liste_mean = ['mean']
    liste_median = ['median']
    liste_var = ['var']
    liste_std = ['std']
    liste_skew = ['skew']
    liste_kurtosis = ['kurtosis']
    liste_mode = ['mode']
    liste_cols = ['Desc']
    liste_max = ['Max']
    liste_min = ['Min']

    for col in liste_variables:
        liste_mean.append(dataframe[col].mean())
        liste_median.append(dataframe[col].median())
        liste_var.append(dataframe[col].var(ddof=0))
        liste_std.append(dataframe[col].std(ddof=0))
        liste_skew.append(dataframe[col].skew())
        liste_kurtosis.append(dataframe[col].kurtosis())
        liste_cols.append(col)
        liste_mode.append(dataframe[col].mode().to_string())
        liste_min.append(dataframe[col].min())
        liste_max.append(dataframe[col].max())

    data_stats = [liste_mean, liste_median, liste_var, liste_std, liste_skew,
                  liste_kurtosis, liste_mode, liste_min, liste_max]
    df_stat = pd.DataFrame(data_stats, columns=liste_cols)

    return df_stat.style.hide_index()

# --------------------------------------------------------------------
# -- PLAGE DE VALEURS MANQUANTES
# --------------------------------------------------------------------


def distribution_variables_plages_perc_donnees( dataframe, variable, liste_bins):
    """
    Retourne les plages des pourcentages des valeurs pour le découpage transmis
    Parameters
    ----------
    dataframe : DataFrame, obligatoire
    variable  : variable à découper obligatoire
    liste_bins: liste des découpages facultatif int ou pintervallindex
    
    sortie : dataframe des plages de nan
    """
    nb_lignes = len(dataframe[variable])
    s_gpe_cut = pd.cut(
        dataframe[variable],
        bins=liste_bins).value_counts().sort_index()
    df_cut = pd.DataFrame({'Plage': s_gpe_cut.index,
                           'nb_données': s_gpe_cut.values})
    df_cut['%_données'] = [
        (row * 100) / nb_lignes for row in df_cut['nb_données']]

    return df_cut.style.hide_index()

# --------------------------------------------------------------------

# --------------------------------------------------------------------
# -- WORDCLOUD + TABLEAU DE FREQUENCE
# --------------------------------------------------------------------


def affiche_wordcloud_tabfreq(
        dataframe,
        variable,
        nom,
        affword=True,
        affgraph=True,
        afftabfreq=True,
        nb_lignes=10):
    """
    Affiche les 'noms' les plus fréquents (wordcloud) et le tableau de fréquence (10 1ères lignes)
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : variable dont on veut voir la fréquence obligatoire
                nom : text affiché dans le tableau de fréquence obligatoire
                nb_lignes : nombre de lignes affichées dans le tab des fréquences facultatives
                affword : booléen : affiche le wordcloud ?
                affgraph : booléen : affiche le graphique de répartition en pourcentage ?
                afftabfreq : booléen : affiche le tableau des fréquences ?
    @param OUT : None
    """
    # Préparation des variables de travail
    dico = dataframe.groupby(variable)[variable].count(
    ).sort_values(ascending=False).to_dict()
    col1 = 'Nom_' + nom
    col2 = 'Nbr_' + nom
    col3 = 'Fréquence (%)'
    df_gpe = pd.DataFrame(dico.items(), columns=[col1, col2])
    df_gpe[col3] = (df_gpe[col2] * 100) / len(dataframe)
    df_gp_red = df_gpe.head(nb_lignes)

    if affword:
        # affiche le wordcloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=100).generate_from_frequencies(dico)
        plt.figure(figsize=(12, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    if affgraph:
        # Barplot de répartition
        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 4))
        sns.barplot(
            y=df_gp_red[col1],
            x=df_gp_red[col3],
            data=df_gp_red,
            color='SteelBlue')
        plt.title('Répartition du nombre de ' + nom)
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    if afftabfreq:
        # affiche le tableau des fréquences
        display(df_gp_red.style.hide_index())

# ---------------------------------------------------------------------------

# --------------------------------------------------------------------
# -- ACP
# --------------------------------------------------------------------

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(8,8))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='11', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,6))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='8', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

def plot_dendrogram(Z, names):
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()
#------------------------------------------------------------------------------


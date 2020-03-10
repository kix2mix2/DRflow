#!/usr/bin/env python
# coding: utf-8

# In[47]:


#Applying Dimensionality Reduction on a csv file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import seaborn as sn; sn.set()
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
from sklearn.manifold import Isomap as ISM
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import SpectralEmbedding as SE
from sklearn.random_projection import GaussianRandomProjection as GRP

#final method which applys standardization and DR on your csv 
#type ist the chosen dr, insert the type as string
#first columns mostly are labels, which need to be dropped before standardization
#Insert the labelnames of the columns as string
def apply_dr(path,type,label= None,label2 = None):
    rawData = pd.read_csv(path)
    stData = standardize_Data(path,label, label2)

    label= rawData[label]
    #label2=rawData[label2]
    #print(stData)
    if type == 'pca':
        data = pca_data(stData,label, label2)
        data.to_csv('pca_data.csv', index = False, header=True)
    elif type == 'tSNE':
        data = tSNE_data(stData,label, label2)
        data.to_csv('tSNE_data.csv', index = False, header=True)
    elif type == 'mds':
        data = mds_data(stData,label, label2)
        data.to_csv('mds_data.csv', index = False, header=True)
    elif type == 'isomap':
        data = isomap_data(stData,label, label2)
        data.to_csv('isomap_data.csv', index = False, header=True)
    elif type == 'laplacian':
        data = se_data(stData,label, label2)
        data.to_csv('laplacian_data.csv', index = False, header=True)
    elif type == 'randprojection':
        data = randProG_data(stData,label, label2)
        data.to_csv('randomProjection_data.csv', index = False, header=True)
    elif type == 'lle':
        data = lle_data(stData,label, label2)
        data.to_csv('lle_data.csv', index = False, header=True)

#Standardiziation of the data. 
#First columns mostly are labels, which need to be dropped before standardization. Insert the labelnames of the columns as string
#Only the first 15000 points are choosen, insert a diffrent number, if you want to calculate less, more or all points (depending on the total point numbers)
def standardize_Data(path, labelName = None, labelName2 = None ):
    rawData = pd.read_csv(path)
    #print(rawData)
    data =rawData
    #if(labelName != None):
    label = rawData[labelName]
    data = rawData.drop([labelName],axis=1)

    standardized_data = StandardScaler().fit_transform(data)
    print(standardized_data.shape)
    return standardized_data

#PCA
def pca_data(stData, labels = None,label2=None):
#     data_1000 = stData[0:1000,:]
#     labels_1000 = labels[0:1000]
    # initializing the pca
    pca = decomposition.PCA()
    # configuring the parameteres
    # the number of components = 2
    pca.n_components = 2
    pca_data = pca.fit_transform(stData)
    # attaching the label for each 2-d data point 
    
    pca_data = np.vstack((pca_data.T, labels)).T
    # creating a new data fram which help us in ploting the result data
    pca_df = pd.DataFrame(data=pca_data, columns=("Dim1", "Dim2","name"))
    #.add_legend() removed, for better vision
#     sn.FacetGrid(pca_df, hue="label", height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
#     plt.show()
    return pca_df

#tSNE
def tSNE_data(standardizeData, labels = None,label2=None):
    # Picking the top 1000 points as TSNE takes a lot of time for 15K points
#     data_1000 = standardizeData[0:1000,:]
#     labels_1000 = labels[0:1000]

    model = TSNE(n_components=2, random_state=0)

    tsne_data = model.fit_transform(standardizeData)
    
    tsne_data = np.vstack((tsne_data.T, labels)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim1", "Dim2","name"))

    #.add_legend() removed, for better vision
#     sn.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, 'Dim1', 'Dim2')
#     plt.show()
    return tsne_df

#Isomap
def isomap_data(stData, labels= None,label2=None):
    # Picking the top 1000 points for faster calculation
#     data_1000 = stData[0:1000,:]
#     labels_1000 = label1[0:1000]

    model = ISM(n_neighbors=5, n_components=2)
    ism_data = model.fit_transform(stData)
    ism_data = np.vstack((ism_data.T, labels)).T

    ism_df = pd.DataFrame(data=ism_data, columns=("Dim1", "Dim2","name"))
    
    #.add_legend() removed, for better vision
#     sn.FacetGrid(ism_df, hue="label", height=6).map(plt.scatter, 'Dim1', 'Dim2')
#     plt.show()
    return ism_df

#Multidimensional scaling
def mds_data(stData, labels= None,label2=None):
    # Picking the top 1000 points for faster calculation
#     data_1000 = stData[0:1000,:]
#     labels_1000 = label1[0:1000]

    model = MDS(n_components=2)
    mds_data = model.fit_transform(stData)
    mds_data = np.vstack((mds_data.T, labels)).T

    mds_df = pd.DataFrame(data=mds_data, columns=("Dim1", "Dim2","name"))
    
    #.add_legend() removed, for better vision
#     sn.FacetGrid(mds_df, hue="label", height=6).map(plt.scatter, 'Dim1', 'Dim2')
#     plt.show()
    return mds_df

#LocallyLinearEmbedding
def lle_data(stData, labels= None,label2=None):
    # Picking the top 1000 points for faster calculation
#     data_1000 = stData[0:1000,:]
#     labels_1000 = label1[0:1000]

    model = LLE(n_neighbors=5, n_components=2)
    lle_data = model.fit_transform(stData)
    lle_data = np.vstack((lle_data.T, labels)).T

    lle_df = pd.DataFrame(data=lle_data, columns=("Dim1", "Dim2","name"))
    #.add_legend() removed, for better vision
#     sn.FacetGrid(lle_df, hue="label", height=6).map(plt.scatter, 'Dim1', 'Dim2')
#     plt.show()
    return lle_df

#Laplacian eigenmap/Spectral Embedding
def se_data(stData, labels= None,label2=None):
    # Picking the top 1000 points for faster calculation
#     data_1000 = stData[0:1000,:]
#     labels_1000 = label1[0:1000]

    model = SE(n_components=2)
    se_data = model.fit_transform(stData)
    se_data = np.vstack((se_data.T, labels)).T

    se_df = pd.DataFrame(data=se_data, columns=("Dim1", "Dim2","name"))
    #.add_legend() removed, for better vision
#     sn.FacetGrid(se_df, hue="label", height=6).map(plt.scatter, 'Dim1', 'Dim2')
#     plt.show()
    return se_df
    
#Random Gaussian projection
def randProG_data(stData, labels= None,label2=None):
    # Picking the top 1000 points for faster calculation
#     data_1000 = stData[0:1000,:]
#     labels_1000 = label1[0:1000]

    model = GRP(n_components=2)
    grp_data = model.fit_transform(stData)
    grp_data = np.vstack((grp_data.T, labels)).T

    grp_df = pd.DataFrame(data=grp_data, columns=("Dim1", "Dim2", "name"))
    #.add_legend() removed, for better vision
#     sn.FacetGrid(grp_df, hue="label", height=6).map(plt.scatter, 'Dim1', 'Dim2')
#     plt.show()
    return grp_df
    
#Sample application
# apply_dr('yalefaces/yalefaces_fullDataset.csv','pca', 'label')

# d0 = pd.read_csv('yalefaces/yalefaces_fullDataset.csv')
# print(d0.head(10))
# print(d0.shape)

# d0 = pd.read_csv('pca_data.csv')
# print(d0.head(10))
# print(d0.shape)


# In[ ]:





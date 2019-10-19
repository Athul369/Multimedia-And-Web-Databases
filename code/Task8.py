import glob
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF


def metadataRead(filepath, imagepath):
    metadataFile = pd.read_csv(filepath, sep=',', header=None)
    metadataFile = metadataFile.iloc[1:]
    metadataFile.columns = ['id', 'age', 'gender', 'skinColor', 'accessories', 'nailPolish', 'aspectOfHand',
                            'imageName', 'irregularities']
    # print(metadataFile)
    imageList = []
    a = 0
    for image in glob.glob(os.path.join(imagepath, "*.jpg")):
        a = a + 1
        imageList.append(image[-16:])

    metadataFile = metadataFile[metadataFile['imageName'].isin(imageList)]
    # print(metadataFile)
    # print(a)
    metadataFile[['aspect', 'orientation']] = metadataFile.aspectOfHand.str.split(expand=True)
    # print(metadataFile)
    metadataFile = metadataFile[['imageName', 'aspect', 'orientation', 'gender', 'accessories']]
    # print(metadataFile)
    convertToBinaryMatrix(metadataFile)


def convertToBinaryMatrix(metaDataFrame):
    a = 0
    metaDataFrame['orientation'] = metaDataFrame['orientation'].replace(['left', 'right'], [0, 1])
    metaDataFrame['aspect'] = metaDataFrame['aspect'].replace(['dorsal', 'palmar'], [0, 1])
    metaDataFrame['gender'] = metaDataFrame['gender'].replace(['male', 'female'], [0, 1])
    # print(metaDataFrame)
    imageList = metaDataFrame['imageName'].tolist()
    # print(imageList)
    metaDataFrame = metaDataFrame[['aspect', 'orientation', 'gender', 'accessories']]
    # print(metaDataFrame)
    metaDataFrame['accessories'] = metaDataFrame['accessories'].astype(int)
    featureList = metaDataFrame.columns
    performNMF(metaDataFrame, 20, imageList, featureList)


def performNMF(metaDataFrame, k, imageList, featureList):
    a = 0
    b = 0
    metaDataMatrix = metaDataFrame.to_numpy()
    # print(metaDataMatrix)
    nmf_ = NMF(n_components=k, init='random', random_state=0)
    W = nmf_.fit_transform(metaDataMatrix)
    H = nmf_.components_
    # print(W)
    # print(H)
    W = rescaleToBasis(W)
    print("Top {} latent semantics in image-space".format(k))
    for i in range(k):
        col = W[:, i]
        arr = []
        for j, val in enumerate(col):
            arr.append((str(imageList[j]), val))
        arr.sort(key=lambda x: x[1], reverse=True)
        print("Printing latent Semantic {} in image-space:".format(i + 1))
        print(arr)
    print("Top {} latent semantics in metadata-space".format(k))
    for i in range(k):
        print(i)
        row = H[i]
        arr = []
        for j, val in enumerate(row):
            arr.append((str(featureList[j]), val))
        arr.sort(key=lambda x: x[1], reverse=True)
        print("Printing latent Semantic {} in metadata-space:".format(i + 1))
        print(arr)


def rescaleToBasis(arr):
    np.seterr(divide='ignore', invalid='ignore')
    row_magnitude = np.sqrt(np.sum(np.square(arr), axis=1))
    rescaled_array = np.divide(arr, row_magnitude[:, None])
    return rescaled_array

#Use 11k metadata set

metadatapath=input("Enter Metadata Path: ")
imagedatapath=input("Enter Image Dataset Path: ")
metadataRead(metadatapath,imagedatapath)


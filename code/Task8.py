import glob
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import Visualizer as vz

def metadataRead(filepath, imagepath, k):
    metadataFile = pd.read_csv(filepath, sep=',', header=None)
    metadataFile = metadataFile.iloc[1:]
    # metadataFile.columns = ['id', 'age', 'gender', 'skinColor', 'accessories', 'nailPolish', 'aspectOfHand',
    #                         'imageName', 'irregularities']
    metadataFile.columns = ['id', 'age', 'gender', 'skinColor', 'accessories', 'nailPolish', 'aspectOfHand',
                            'Orientation', 'imageName', 'irregularities']
    # print(metadataFile)
    imageList = []
    a = 0
    for image in glob.glob(os.path.join(imagepath, "*.jpg")):
        a = a + 1
        imageList.append(image[-16:])

    metadataFile = metadataFile[metadataFile['imageName'].isin(imageList)]
    # print(metadataFile)
    # print(a)

    # metadataFile[['aspect', 'orientation']] = metadataFile.aspectOfHand.str.split(expand=True)

    # print(metadataFile)
    # metadataFile = metadataFile[['imageName', 'aspect', 'orientation', 'gender', 'accessories']]

    metadataFile = metadataFile[['imageName', 'aspectOfHand', 'Orientation', 'gender', 'accessories']]
    # print(metadataFile)
    convertToBinaryMatrix(metadataFile, k)


def convertToBinaryMatrix(metaDataFrame, k):
    a = 0
    # metaDataFrame['orientation'] = metaDataFrame['orientation'].replace(['left', 'right'], [0, 1])
    metaDataFrame['Orientation'] = metaDataFrame['Orientation'].replace(['left', 'right'], [0, 1])
    # metaDataFrame['aspect'] = metaDataFrame['aspect'].replace(['dorsal', 'palmar'], [0, 1])
    metaDataFrame['aspectOfHand'] = metaDataFrame['aspectOfHand'].replace(['dorsal', 'palmar'], [0, 1])
    metaDataFrame['gender'] = metaDataFrame['gender'].replace(['male', 'female'], [0, 1])
    # print(metaDataFrame)
    imageList = metaDataFrame['imageName'].tolist()
    # print(imageList)
    # metaDataFrame = metaDataFrame[['aspect', 'orientation', 'gender', 'accessories']]
    metaDataFrame = metaDataFrame[['aspectOfHand', 'Orientation', 'gender', 'accessories']]
    # print(metaDataFrame)
    metaDataFrame['accessories'] = metaDataFrame['accessories'].astype(int)
    featureList = metaDataFrame.columns
    performNMF(metaDataFrame, k, imageList, featureList)


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
    img_space = []
    print("Top {} latent semantics in image-space".format(k))
    for i in range(k):
        col = W[:, i]
        arr = []
        for j, val in enumerate(col):
            arr.append((str(imageList[j]), val))
        arr.sort(key=lambda x: x[1], reverse=True)
        print("Printing latent Semantic {} in image-space:".format(i + 1))
        print(arr)
        img_space.append(arr[:50])
    img_space = pd.DataFrame(img_space)
    print(img_space.shape)
    vz.visualize_img_space(k, img_space)
    print("Top {} latent semantics in metadata-space".format(k))
    metadata_space = []
    for i in range(k):
        print(i)
        row = H[i]
        arr = []
        for j, val in enumerate(row):
            arr.append((str(featureList[j]), val))
        arr.sort(key=lambda x: x[1], reverse=True)
        print("Printing latent Semantic {} in metadata-space:".format(i + 1))
        print(arr)
        metadata_space.append(arr)
    metadata_space = pd.DataFrame(metadata_space)
    vz.visualize_metadata_space(k, metadata_space)


def rescaleToBasis(arr):
    np.seterr(divide='ignore', invalid='ignore')
    row_magnitude = np.sqrt(np.sum(np.square(arr), axis=1))
    rescaled_array = np.divide(arr, row_magnitude[:, None])
    return rescaled_array


#Use 11k metadata set
# "C:\Users\tyler\Documents\Xfer to ASU Google Drive\CSE 515\Project\11k Hands Data\Metadata\HandInfo.csv"
# metadatapath = input("Enter Metadata Path: ")
# "C:\Users\tyler\Documents\Xfer to ASU Google Drive\CSE 515\Project\11k Hands Data\Hands"
# imagedatapath = input("Enter Image Dataset Path: ")

# metadataRead(r"C:\Users\tyler\Documents\Xfer to ASU Google Drive\CSE 515\Project\11k Hands Data\Metadata\HandInfo.csv",
#              r"C:\Users\tyler\Documents\Xfer to ASU Google Drive\CSE 515\Project\11k Hands Data\Hands")

def run_task_8(k):
    metadataRead(r"C:\Users\tyler\Desktop\phase2test\cse515_dataset_2\Dataset2\ImageMetadata.csv",
                 r"C:\Users\tyler\Desktop\phase2test\cse515_dataset_2\Dataset2", k)


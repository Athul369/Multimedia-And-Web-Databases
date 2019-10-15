from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import pymongo
import os
from numpy import linalg
import math
import shutil

client = pymongo.MongoClient('localhost', 27017)
imagedb = client["imagedb"]
mydb = imagedb["image_models"]
meta =  imagedb["ImageMetadata"]

class PrincipleComponentAnalysis(object):

    def createPCA_KLatentSemantics(self, model, k):
        #svd1 = TruncatedSVD(k)
        model = "bag_" + model
        frames = []
        img_list = []
        for descriptor in imagedb.image_models.find():
            frames.append(descriptor[model])
            img_list.append(descriptor["_id"])

        frames =pd.DataFrame(frames)
        print(frames.shape)
        mean_vec = np.mean(frames, axis=0)
        cov_mat = np.cov(frames.T)
        print(cov_mat.shape)

        # Compute the eigen values and vectors using numpy
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        y = 0
        for p in eig_pairs:
            feature = pd.DataFrame(p[1])
            y += 1
            if y == 1:
                frame = feature
            else:
                frame = [frame, feature]
                frame = pd.concat(frame, axis=1, sort=False)

        for i in range(k):
            col = frame.iloc[:, i]
            arr = []
            for k, val in enumerate(col):
                arr.append((k, val))
            arr.sort(key=lambda x: x[1], reverse=True)
            print("Printing term-weight pair for latent Symantic L{}:".format(i + 1))
            print(arr)


    def mSimilarImage(self, imgLoc, model, k, m):
        img_list = []
        pca = PCA(k)
        model = "bag_" + model
        feature_desc = []
        img_list = []
        for descriptor in imagedb.image_models.find():
            feature_desc.append(descriptor[model])
            img_list.append(descriptor["_id"])

        feature_desc_transformed = pca.fit_transform(feature_desc)

        head, tail = os.path.split(imgLoc)

        id = img_list.index(tail)

        rank_dict = {}
        for i, row in enumerate(feature_desc_transformed):
            if (i == id):
                continue
            euc_dis = np.square(np.subtract(feature_desc_transformed[id], feature_desc_transformed[i]))
            match_score = np.sqrt(euc_dis.sum(0))
            rank_dict[img_list[i]] = match_score

        # res_dir = os.path.join('..', 'output', model[4:], 'match')
        # if os.path.exists(res_dir):
        #     shutil.rmtree(res_dir)
        # os.mkdir(res_dir)
        count = 0
        print("\n\nNow printing top {} matched Images and their matching scores".format(m))
        for key, value in sorted(rank_dict.items(), key=lambda item: item[1]):
            if count < m:
                print(key + " has matching score:: " + str(value))
                # shutil.copy(os.path.join(head, key), res_dir)
                count += 1
            else:

                break

    def LabelLatentSemantic(self, label, model, k):

        model = "bag_" + model
        if label ==  "left" or label == "right":
            search = "Orientation"
        elif label == "dorsal" or label == "palmar":
            search = "aspectOfHand"
        elif label == "Access" or label == "NoAccess":
            search = "accessories"
        elif label == "male" or label == "female":
            search = "gender"
        else:
            print("Please provide correct label")
            exit(1)

        frames = []
        img_list = []
        imageslist_Meta = []

        for descriptor in imagedb.ImageMetadata.find():
            if descriptor[search] == label:
                imageslist_Meta.append(descriptor["imageName"])


        for descriptor in imagedb.image_models.find():
            if descriptor["_id"] in imageslist_Meta:
                frames.append(descriptor[model])
                img_list.append(descriptor["_id"])


        frames = pd.DataFrame(frames)

        mean_vec = np.mean(frames, axis=0)
        cov_mat = np.cov(frames.T)


        # Compute the eigen values and vectors using numpy
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        y = 0
        for p in eig_pairs:
            feature = pd.DataFrame(p[1])
            y += 1
            if y == 1:
                frame = feature
            else:
                frame = [frame, feature]
                frame = pd.concat(frame, axis=1, sort=False)

        for i in range(k):
            col = frame.iloc[:, i]
            arr = []
            for k, val in enumerate(col):
                arr.append((k, val))
            arr.sort(key=lambda x: x[1], reverse=True)
            print("Printing term-weight pair for latent Symantic L{}:".format(i + 1))
            print(arr)

    def mSimilarImage_Label(self, imgLoc,label,model, k, m):

        if label == "left" or label == "right":
            search = "Orientation"
        elif label == "dorsal" or label == "palmar":
            search = "aspectOfHand"
        elif label == "Access" or label == "NoAccess":
            search = "accessories"
            if label == "Access":
                label = 1
            else:
                label = 0

        elif label == "male" or label == "female":
            search = "gender"
        else:
            print("Please provide correct label")
            exit(1)

        pca = PCA(k)
        model = "bag_" + model
        img_list = []
        imageslist_Meta = []
        frames = []

        for descriptor in imagedb.ImageMetadata.find():
            if descriptor[search] == label:
                imageslist_Meta.append(descriptor["imageName"])

        print(len(imageslist_Meta))

        for descriptor in imagedb.image_models.find():
            if descriptor["_id"] in imageslist_Meta:
                frames.append(descriptor[model])
                img_list.append(descriptor["_id"])

        feature_desc_transformed = pca.fit_transform(frames)

        head, tail = os.path.split(imgLoc)

        id = img_list.index(tail)

        rank_dict = {}
        for i, row in enumerate(feature_desc_transformed):
            if (i == id):
                continue
            euc_dis = np.square(np.subtract(feature_desc_transformed[id], feature_desc_transformed[i]))
            match_score = np.sqrt(euc_dis.sum(0))
            rank_dict[img_list[i]] = match_score

        # res_dir = os.path.join('..', 'output', model[4:], 'match')
        # if os.path.exists(res_dir):
        #     shutil.rmtree(res_dir)
        # os.mkdir(res_dir)
        count = 0
        print("\n\nNow printing top {} matched Images and their matching scores".format(m))
        for key, value in sorted(rank_dict.items(), key=lambda item: item[1]):
            if count < m:
                print(key + " has matching score:: " + str(value))
                # shutil.copy(os.path.join(head, key), res_dir)
                count += 1
            else:

                break

    def ImageClassfication(self, imgLoc, model, k ):

        result = {}
        model = "bag_" + model
        head, tail = os.path.split(imgLoc)
        query_desc = []
        for descriptor in imagedb.image_models.find():
            if descriptor["_id"] == tail:
                query_desc.append(descriptor[model])

        labels = [ "dorsal", "palmar", "left", "right", "Access", "NoAccess", "male", "female"]

        for label in labels:
            # print(label)

            if label == "left" or label == "right":
                search = "Orientation"
            elif label == "dorsal" or label == "palmar":
                search = "aspectOfHand"
            elif label == "Access" or label == "NoAccess":
                search = "accessories"
                if label == "Access":
                    label = 1
                else:
                    label = 0
            elif label == "male" or label == "female":
                search = "gender"
            else:
                print("Please provide correct label")
                exit(1)

            img_list = []
            imageslist_Meta = []
            frames = []

            for descriptor in imagedb.ImageMetadata.find():
                if search == "Orientation" and descriptor["aspectOfHand"] == "palmar":
                    if descriptor[search] != label:
                        imageslist_Meta.append(descriptor["imageName"])
                    continue
                if descriptor[search] == label:
                    imageslist_Meta.append(descriptor["imageName"])

            # print(len(imageslist_Meta))

            for descriptor in imagedb.image_models.find():
                if descriptor["_id"] in imageslist_Meta:
                    # if descriptor["_id"] == tail:
                    #     continue
                    frames.append(descriptor[model])
                    img_list.append(descriptor["_id"])

            pca = PCA(k)
            pca_Obj = pca.fit(frames)
            feature_desc_transformed = pca_Obj.transform(frames)

            mean_transformed = np.true_divide(feature_desc_transformed.sum(0),len(img_list))
            # print(mean_transformed)
            query_desc_transformed = pca_Obj.transform(query_desc)

            all_dist = []


            # for D_des in (feature_desc_transformed):
            distance = np.linalg.norm(mean_transformed - query_desc_transformed)
                    # all_dist.append(distance)
            # min_dist = min(all_dist)
            result[label] = distance

        flag = False
        if result["dorsal"] > result["palmar"]:
            flag = True
            print("palmar")
        else:
            print("dorsal")
        if result["left"] > result["right"]:
            if flag:
                print("Left")
            else:
                print("Right")
        else:
            if flag:
                print("Right")
            else:
                print("Left")

        if result[1] > result[0]:
            print("NoAccess")
        else:
            print("Access")

        if result["male"] > result["female"]:
            print("female")
        else:
            print("male")


    def BOW(self, model):
        model = "bag_" + model
        labels = ["left", "right", "dorsal", "palmar", "Access", "NoAccess", "male", "female"]

        for label in labels:
            # print(label)

            if label == "left" or label == "right":
                search = "Orientation"
            elif label == "dorsal" or label == "palmar":
                search = "aspectOfHand"
            elif label == "Access" or label == "NoAccess":
                search = "accessories"
                if label == "Access":
                    label = 1
                else:
                    label = 0
            elif label == "male" or label == "female":
                search = "gender"
            else:
                print("Please provide correct label")
                exit(1)

            img_list = []
            imageslist_Meta = []
            frames = []

            for descriptor in imagedb.ImageMetadata.find():
                if descriptor[search] == label:
                    imageslist_Meta.append(descriptor["imageName"])

            # print(len(imageslist_Meta))

            for descriptor in imagedb.image_models.find():
                if descriptor["_id"] in imageslist_Meta:
                    frames.append(descriptor[model])
                    img_list.append(descriptor["_id"])

            frames = pd.DataFrame(frames)
            img_list = pd.DataFrame(img_list)
            data = [img_list, frames]
            frame = pd.concat(data, axis=1, sort=False)
            filepath = "D:/CSE515MultiMediaWebDB/BoW_Analysis/" + str(label) + ".xlsx"
            frame.to_excel(filepath, header=False, index=False)
import glob
import os
import shutil
import pandas as pd
import numpy as np
import pymongo
import Visualizer as vz
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

client = pymongo.MongoClient('localhost', 27018)
imagedb = client["imagedb"]
mydb = imagedb["image_models"]


class NM_F(object):

    def createKLatentSymantics(self, model, k):

        model_name = model
        model = "bag_" + model
        feature_desc = []
        img_list = []
        for descriptor in imagedb.image_models.find():
            feature_desc.append(descriptor[model])
            img_list.append(descriptor["_id"])
        model = NMF(n_components=k, init='random', random_state=0)
        W = model.fit_transform(feature_desc)
        H = model.components_
        W=NM_F.rescaleToBasis(W)

        visualizeArr = []

        for i in range(k):
            col = W[:, i]
            arr = []
            for k, val in enumerate(col):
                arr.append((str(img_list[k]), val))
            arr.sort(key=lambda x: x[1], reverse=True)
            """ Only take the top 5 data objects to report for each latent semantic """
            visualizeArr.append(arr[:5])
            print("Printing term-weight pair for latent Semantic {}:".format(i + 1))
            print(arr)
        visualizeArr = pd.DataFrame(visualizeArr)
        vz.visualize_data_ls(visualizeArr, 'NMF', model_name)
        print(W)
        
        #####Feature discriptor and latent space dot product . below "feature_latent_product" functions returns a array

        feature_latentsemantics_visualizer=NM_F.feature_latent_product(feature_desc,H,img_list)
        print(feature_latentsemantics_visualizer)
        

    def mSimilarImage(self, imgLoc, model, k, m):
        model_name = model

        model = "bag_" + model
        feature_desc = []
        img_list = []
        for descriptor in imagedb.image_models.find():
            feature_desc.append(descriptor[model])
            img_list.append(descriptor["_id"])
        nmf_ = NMF(n_components=k)
        W = nmf_.fit_transform(feature_desc)
        H = nmf_.components_

        head, tail = os.path.split(imgLoc)

        id = img_list.index(tail)

        rank_dict = {}
        for i, row in enumerate(W):
            if (i == id):
                continue
#             euc_dis = np.square(np.subtract(W[id], W[i]))
#             match_score = np.sqrt(euc_dis.sum(0))
#             rank_dict[img_list[i]] = match_score
            match_score=NM_F.nvsc(W[id], W[i])
            print(match_score)
            rank_dict[img_list[i]]=match_score
        # res_dir = os.path.join('..', 'output', model[4:], 'match')
        # if os.path.exists(res_dir):
        #     shutil.rmtree(res_dir)
        # os.mkdir(res_dir)

        count = 0
        print("\n\nNow printing top {} matched Images and their matching scores".format(m))
        # sorted_dict = sorted(rank_dict.items(), key=lambda item: item[1])
        head, tail = os.path.split(imgLoc)
        vz.visualize_matching_images(tail, rank_dict, m, 'NMF', model_name)
        for key, value in sorted(rank_dict.items(), key=lambda item: item[1]):
            if count < m:
                print(key + " has matching score:: " + str(value))
                # shutil.copy(os.path.join(head, key), res_dir)
                count += 1
            else:
                break

    def LabelLatentSemantic(self, label, model, k):
        a = 0
        model = "bag_" + model
        if label == "left" or label == "right":
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

        feature_desc = []
        img_list = []
        imageslist_Meta = []

        for descriptor in imagedb.ImageMetadata.find():
            if descriptor[search] == label:
                imageslist_Meta.append(descriptor["imageName"])

        print(len(imageslist_Meta))

        for descriptor in imagedb.image_models.find():
            if descriptor["_id"] in imageslist_Meta:
                feature_desc.append(descriptor[model])
                img_list.append(descriptor["_id"])

        print(len(img_list))

        nmf_ = NMF(n_components=k, init='random', random_state=0)


        feature_desc_transformed = nmf_.fit_transform(feature_desc)
        #H = nmf_.components_
        W=NM_F.rescaleToBasis(feature_desc_transformed)



        for i in range(k):
            col = W[:, i]
            arr = []
            for k, val in enumerate(col):
                arr.append((str(img_list[k]), val))
            arr.sort(key=lambda x: x[1], reverse=True)
            print("Printing term-weight pair for latent Symantic {}:".format(i + 1))
            print(arr)

        return feature_desc_transformed

    def mSimilarImage_Label(self, imgLoc, label, model, k, m):

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

        # svd = TruncatedSVD(k)
        nmf_ = NMF(n_components=k, init='random', random_state=0)
        model = "bag_" + model
        img_list = []
        imageslist_Meta = []
        feature_desc = []

        for descriptor in imagedb.ImageMetadata.find():
            if descriptor[search] == label:
                imageslist_Meta.append(descriptor["imageName"])

        print(len(imageslist_Meta))

        for descriptor in imagedb.image_models.find():
            if descriptor["_id"] in imageslist_Meta:
                feature_desc.append(descriptor[model])
                img_list.append(descriptor["_id"])

        feature_desc_transformed = nmf_.fit_transform(feature_desc)

        head, tail = os.path.split(imgLoc)

        id = img_list.index(tail)

        rank_dict = {}
        for i, row in enumerate(feature_desc_transformed):
            if (i == id):
                continue
#             euc_dis = np.square(np.subtract(feature_desc_transformed[id], feature_desc_transformed[i]))
#             match_score = np.sqrt(euc_dis.sum(0))
#             rank_dict[img_list[i]] = match_score
            match_score = NM_F.nvsc(feature_desc_transformed[id], feature_desc_transformed[i])
            print(match_score)
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
                #shutil.copy(os.path.join(head, key), res_dir)
                count += 1
            else:
                break

    def ImageClassfication(self, imgLoc, model, k):
        result = {}
        model = "bag_" + model
        head, tail = os.path.split(imgLoc)
        query_desc = []
        for descriptor in imagedb.image_models.find():
            if descriptor["_id"] == tail:
                query_desc.append(descriptor[model])

        labels = ["dorsal", "palmar", "left", "right", "Access", "NoAccess", "male", "female"]

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
                if descriptor["_id"] in imageslist_Meta and descriptor["_id"] != tail:
                    frames.append(descriptor[model])
                    img_list.append(descriptor["_id"])

            # svd = TruncatedSVD(k)
            nmf_ = NMF(n_components=k, init='random', random_state=0)
            feature_desc_transformed = nmf_.fit_transform(frames)
            query_desc_transformed = nmf_.transform(query_desc)
            mean_transformed = np.true_divide(feature_desc_transformed.sum(0), len(img_list))

            all_dist = []

            # for D_des in feature_desc_transformed:
            #     distance = np.linalg.norm(D_des - query_desc_transformed)
            #     all_dist.append(distance)
            # min_dist = min(all_dist)

            min_dist = np.linalg.norm(mean_transformed - query_desc_transformed)

            result[label] = min_dist

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

    def rescaleToBasis(arr):
        a = 0
        col_magnitude=np.sqrt(np.sum(np.square(arr), axis=0))
        rescaled_array=np.divide(arr,col_magnitude)
        return rescaled_array
    
    ################### distance measurement function
    def nvsc(X, Y):
        sumMin = 0
        sumMax = 0
        for i in range(len(X)):
            a = 0
            sumMin = sumMin + min(X[i], Y[i])
            sumMax = sumMax + max(X[i], Y[i])
        chi = sumMin / sumMax
        distance = 1 - chi * chi
        return distance
    
####################feature- latentsemantics  visualizer function

    def feature_latent_product(featMat,latMat,image_list):
        a=0
        visualizerDict=[]
        for i in range(len(latMat)):
            maxDict={}

            for j in range(len(featMat)):
                maxDict[image_list[j]]=np.dot(latMat[i],featMat[j])

            maximum = max(maxDict, key=maxDict.get)
            visualizerDict.append((maximum,maxDict[maximum]))

        return visualizerDict





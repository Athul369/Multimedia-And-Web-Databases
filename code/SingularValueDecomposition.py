from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd
import numpy as np
import pymongo
import os
import shutil

client = pymongo.MongoClient('localhost', 27018)
imagedb = client["imagedb"]
mydb = imagedb["image_models"]


class SVD(object):

    def createKLatentSymantics(self, model, k):
        #svd1 = TruncatedSVD(k)
        model = "bag_" + model
        feature_desc = []
        img_list = []
        for descriptor in imagedb.image_models.find():
            feature_desc.append(descriptor[model])
            img_list.append(descriptor["_id"])

        #feature_desc_transformed = svd1.fit_transform(feature_desc)
        U, S, V = svd(feature_desc, full_matrices=False)

        for i in range(k):
            col = U[:, i]
            arr = []
            for k, val in enumerate(col):
                arr.append((str(img_list[k]), val))
            arr.sort(key=lambda x: x[1], reverse=True)
            print("Printing term-weight pair for latent Symantic {}({}):".format(i + 1, S[i]))
            print(arr)

    def mSimilarImage(self, imgLoc, model, k, m):
        img_list = []
        svd = TruncatedSVD(k)
        model = "bag_" + model
        feature_desc = []
        img_list = []
        for descriptor in imagedb.image_models.find():
            feature_desc.append(descriptor[model])
            img_list.append(descriptor["_id"])

        feature_desc_transformed = svd.fit_transform(feature_desc)

        head, tail = os.path.split(imgLoc)

        id = img_list.index(tail)

        rank_dict = {}
        for i, row in enumerate(feature_desc_transformed):
            if (i == id):
                continue
            euc_dis = np.square(np.subtract(feature_desc_transformed[id], feature_desc_transformed[i]))
            match_score = np.sqrt(euc_dis.sum(0))
            rank_dict[img_list[i]] = match_score

        res_dir = os.path.join('..', 'output', model[4:], 'match')
        if os.path.exists(res_dir):
            shutil.rmtree(res_dir)
        os.mkdir(res_dir)
        count = 0
        print("\n\nNow printing top {} matched Images and their matching scores".format(m))
        for key, value in sorted(rank_dict.items(), key=lambda item: item[1]):
            if count < m:
                print(key + " has matching score:: " + str(value))
                shutil.copy(os.path.join(head, key), res_dir)
                count += 1
            else:
                break

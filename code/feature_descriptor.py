from LocalBinaryPatterns import LBP
from ColorMoments import CM
from SIFT import SIFT
from HOGmain import HOG
import os
import glob
import pymongo
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# import dbtask

client = pymongo.MongoClient('localhost', 27018)
imagedb = client["imagedb"]
mydb = imagedb["image_models"]

def createKMeans(model, k):
    feature_desc = None
    for descriptor in imagedb.image_models.find():
        if feature_desc is None:
            feature_desc=pd.DataFrame(descriptor[model])
        else:
            feature_desc=[feature_desc, pd.DataFrame(descriptor[model])]
            feature_desc = pd.concat(feature_desc, axis=0, sort=False)

    feature_desc = feature_desc.values
    ret = KMeans(n_clusters=k, max_iter=1000).fit(feature_desc)

    for item in imagedb.image_models.find():
        img=pd.DataFrame(item[model])
        x=ret.predict(img)
        bag = np.zeros((k,), dtype=int)
        for z in x:
            bag[z-1]+=1
        imageID = item["_id"]
        bag = bag.tolist()
        imagedb.image_models.update_one({"_id" : imageID}, {"$set": {"bag_"+model : bag}})

def calculate_fd(path):
    for image in glob.glob(os.path.join(path, "*.jpg")):
        dict = {}
        dict["_id"] = image[-16:]

        md = CM(image)
        lst = md.getFeatureDescriptors()
        print(lst)
        dict["CM"] = lst

        md = LBP(image)
        lst = md.getFeatureDescriptors()
        dict["LBP"] = lst

        md = SIFT(image)
        lst = md.getFeatureDescriptors()
        dict["SIFT"] = lst

        md = HOG(image)
        lst = md.getFeatureDescriptors()
        lst = lst.tolist()
        dict["HOG"] = lst

        #dict["HOG"] = lst.tolist()
        #print(type(lst.tolist()))

        rec = imagedb.image_models.insert_one(dict)

## Main
#path = input("Enter Path: ")
#calculate_fd(path)

createKMeans("CM", 30)
createKMeans("SIFT", 30)
createKMeans("LBP", 30)


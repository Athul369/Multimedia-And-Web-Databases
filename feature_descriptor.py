from LocalBinaryPatterns import LBP
from ColorMoments import CM
from SIFT import SIFT
from HOGmain import HOG
import os
import glob
import pymongo

# import dbtask

client = pymongo.MongoClient('localhost', 27017)
imagedb = client["imagedb"]
models = imagedb["image_models"]

def calculate_fd(path):
    for image in glob.glob(os.path.join(path, "*.jpg")):
        dict = {}
        dict["_id"] = image[-16:]

        md = CM(image)
        lst = md.getFeatureDescriptors()
        dict["CM"] = lst

        md = LBP(image)
        lst = md.getFeatureDescriptors()
        dict["LBP"] = lst

        # md = SIFT(image)
        # lst = md.getFeatureDescriptors()
        # dict["SIFT"] = lst

        md = HOG(image)
        lst = md.getFeatureDescriptors()
        lst = lst.tolist()
        dict["HOG"] = lst

        # dict["HOG"] = lst.tolist()
        # print(type(lst.tolist()))

        rec = mydb.image_models.insert_one(dict)

## Main
path = input("Enter Path: ")
calculate_fd(path)

import math
from collections import defaultdict
from functools import reduce
from sklearn.decomposition import TruncatedSVD

from LocalBinaryPatterns import LBP
from ColorMoments import CM
from SIFT import SIFT
from HOGmain import HOG

import numpy as np
import pandas as pd

#from phase1.csvProcessor import CsvProcessor

SEED = 12
np.random.seed(SEED)
IMAGE_ID_COL = 'ImageId'
import pymongo

client = pymongo.MongoClient('localhost', 27017)
imagedb = client["imagedb"]
mydb = imagedb["image_models"]

class LSH:

    def __init__(self, hash_obj, num_layers, num_hash, vec, b, w):
        self.hash_obj = hash_obj
        self.num_layers = num_layers
        self.num_hash = num_hash
        self.vec = vec
        self.b = b
        self.w = w

    def create_hash_table(self, img_vecs, verbose=False):
        """ Vectorized hash function to bucket all img vecs

            Returns
            -------
            hash_table : List of List of defaultdicts
            
        """
        #print(img_vecs)
        #print(type(img_vecs))
        hash_table = self.init_hash_table()
        nrows, ncols = img_vecs.shape[0], img_vecs.shape[1]
        for i in range(nrows):
            img_id, img_vec = img_vecs[i][-1], np.array(img_vecs[i][:-1])
            for idx, hash_vec in enumerate(hash_table):
                buckets = self.hash_obj.hash(img_vec, self.vec[idx], self.b[idx], self.w)
                for i in range(len(buckets)):
                    hash_vec[i][buckets[i]].add(img_id)
        return hash_table

    def init_hash_table(self):
        hash_table = []
        for i in range(self.num_layers):
            hash_layer = []
            for j in range(self.num_hash):
                hash_vec = defaultdict(set)
                hash_layer.append(hash_vec)
            hash_table.append(hash_layer)
        return hash_table

    def find_ann(self, query_point, hash_table, k):
        candidate_imgs = set()
        num_conjunctions = self.num_hash
        for layer_idx, layer in enumerate(self.vec):
            hash_vec = hash_table[layer_idx]
            buckets = self.hash_obj.hash(query_point, layer, self.b[layer_idx], self.w)
            cand = hash_vec[0][buckets[0]].copy()
            # self.test(hash_vec[1])
            for ix, idx in enumerate(buckets[1:num_conjunctions]):
                # needs ix+1 since we already took care of index 0
                cand = cand.intersection(hash_vec[ix + 1][idx])
            candidate_imgs = candidate_imgs.union(cand)
            print("---------------  Candidate Images  ------------------")
            print(candidate_imgs)
            if len(candidate_imgs) > 4 * k:
                print(f'Early stopping at layer {layer_idx} found {len(candidate_imgs) }')
                break
        if len(candidate_imgs) < k:
            if num_conjunctions > 1:
                self.num_hash -= 1
                print('Reduced number of hashes')
                return self.find_ann(query_point, hash_table, k=k)
            else:
                print('Cannot reduce number of hashes')
        return candidate_imgs

    def post_process_filter(self, query_point, candidates, k):
        distances = [{IMAGE_ID_COL: row['ImageID'],
                      'distance': self.hash_obj.dist(query_point, row.drop('ImageID'))}
                     for idx, row in candidates.iterrows()]
        # distances []
        # for row in candidates.iterrows():
        #    dist = self.hash_obj.dist(query_point, )
        return sorted(distances, key=lambda x: x['distance'])[:k]


class l2DistHash:

    def hash(self, point, vec, b, w):
        """
            Parameters
            ----------
            point :
            vec:

            Returns
            -------
            numpy array of which buckets point falls in given layer
        """
        val = np.dot(vec, point) + b
        val = val * 100
        res = np.floor_divide(val, w)
        # print(len(res))
        return res

    def dist(self, point1, point2):
        v = (point1 - point2)**2
        return math.sqrt(sum(v))

final_desc = []
img_df = None

def run_lsh(input_vec, num_layers, num_hash):
    w = 400
    dim = 257
    vec = np.random.rand(num_layers, num_hash, dim - 1)
    b = np.random.randint(low=0, high=w, size=(num_layers, num_hash))
    l2_dist_obj = l2DistHash()
    lsh = LSH(hash_obj=l2_dist_obj, num_layers=num_layers, num_hash=num_hash, vec=vec, b=b, w=w)
    hashTable = lsh.create_hash_table(input_vec, verbose=False)
    return hashTable

def getFeature(query_path, model_name):
    if model_name == "CM":
        fd = CM(query_path)
    elif model_name == "HOG":
        fd = HOG(query_path)
    elif model_name == "SIFT":
        fd = SIFT(query_path)
    elif model_name == "LBP":
        fd = LBP(query_path)
    else:
        print("Error! Invalid Model Name")
        return []
    lst = fd.getFeatureDescriptors()
    return lst

def img_ann(img_df, query, k, num_layers=10, num_hash=10, layer_file_name=None):
    svd = TruncatedSVD(256)
    feature_desc_transformed = svd.fit_transform(np.array(img_df.iloc[:, 1:]))
    image_ids = img_df.iloc[:,0]
    w = 50
    dim = feature_desc_transformed.shape[1]
    feature_desc_transformed = pd.DataFrame(feature_desc_transformed)
    feature_desc_transformed['ImageID'] = image_ids
    
    #print(feature_desc_transformed)
    # Create vector with rand num in num_layers X num_hash X dim-1(1 dim for img_id)
    vec = np.random.rand(num_layers, num_hash, dim)
    #vec = np.arange(num_layers*num_hash*(dim-1)).reshape(num_layers, num_hash, dim-1)
    b = np.random.randint(low=0, high=w, size=(num_layers, num_hash))
    # b = np.arange(num_layers*num_hash).reshape(num_layers, num_hash)
    #print("^^^^^^^^^^^^^^^^^^^^^^^")
    l2_dist_obj = l2DistHash()
    lsh = LSH(hash_obj=l2_dist_obj, num_layers=num_layers, num_hash=num_hash, vec=vec, b=b, w=w)
    hash_table = lsh.create_hash_table(feature_desc_transformed.values)
    #query_vec = getFeature(query, "CM")
    query_vec = feature_desc_transformed.loc[feature_desc_transformed['ImageID'] == query]
    query_vec = query_vec.iloc[:,:-1].values[0]
    print(query_vec)
    #query_vec = query_vec.iloc[0, :-1]
    #print(query_vec)
    # query_vec = feature_desc_transformed[10]
    # t = len(query_vec)
    # query_vec = np.array(query_vec)
    #query_vec = query_vec.values.reshape(t, )
    
    candidate_ids = lsh.find_ann(query_point=query_vec, hash_table=hash_table, k=k)
    candidate_vecs = feature_desc_transformed.loc[feature_desc_transformed['ImageID'].isin(candidate_ids)]
    print(candidate_vecs)
    print(type(candidate_vecs))
    if not candidate_ids:
        return None
    dist_res = lsh.post_process_filter(query_point=query_vec, candidates=candidate_vecs, k=k)
    # for i in dist_res:
    #     img_id = i[0]
    #     i['loc'] = img_id_loc_df.loc[img_id_loc_df[0] == img_id, 'location'].item()
    return dist_res

for descriptor in imagedb.image_models.find():
    desc = []
    desc.append(descriptor['_id'])
    cm_desc = descriptor['CM']
    cm_concat_desc = [i for cm in cm_desc for i in cm]
    for i in cm_concat_desc:
        desc.append(i)
    final_desc.append(desc)
    #print(final_desc)

img_df = pd.DataFrame(final_desc)
#print(img_df)
img_df = img_df.iloc[0:6000, :]
#hash_table, img_df_transformed = run_lsh(img_df, 10, 10)
#print(img_df_transformed)
# = run_lsh(img_df, 100, 20)
#print(len(hash_table[0][0]))
result = img_ann(img_df, 'Hand_0000674.jpg', 20)
print(result)
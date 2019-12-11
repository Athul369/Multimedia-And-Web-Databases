import argparse
import os
import glob
import shutil
from os.path import isfile, join
import pandas as pd
import Constants as const
import image_resize as ir


from Task1 import Task1
from LSH import *
from Task4SVM import SVM
from Task4DecisionTree import DecTree
from PPR import PersonalizedPageRank
from Task2 import Query_input
from feature_descriptor import *
import Visualizer as vz


#Parsing the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir',action="store", dest="dir",help="Provide directory name", default="None")
parser.add_argument('-l', '--label',action="store", dest="label",help="Provide labelled csv name", default="None")
parser.add_argument('-u', '--unlabel',action="store", dest="unlabel",help="Provide unlabelled csv name", default="None")
parser.add_argument('-i', '--imageid',action="store", dest="imageid",help="Provide image name", default="None")
parser.add_argument('-k', '--klatent',type=int, dest="klatent",help="Provide k value to get k latent symantics", default=20)
parser.add_argument('-c', '--centers',type=int, dest="centers",help="Provide centers count", default=10)
parser.add_argument('-L', '--layers',type=int, dest="layers",help="Provide layers count", default=10)
parser.add_argument('-m', '--mimage',type=int, dest="mimage",help="Provide m value to get m similar images", default=10)
parser.add_argument('-t', '--taskid',type=int, dest="taskid", help="Provide the task number", default=-1)
parser.add_argument('-I', '--list', action='store',dest='list',help='Pass the image list', default="None")
parser.add_argument('-T', '--type',action="store", dest="type",help="Provide type of classifier", default="None")


args = parser.parse_args()

if not 0 <= args.taskid <= 5:
    print("Please provide valid task Id using option -t OR --taskid")
    exit(1)

task_id = args.taskid
k = args.klatent
m = args.mimage
c = args.centers

if task_id == 0:
    if args.dir == "None":
        print("Please provide directory name")
        exit(1)
    path = args.dir
    labelled_path = os.path.join(path, 'Labelled')
    unlabelled_path = os.path.join(path, 'Unlabelled')
    mislabelled_path = os.path.join(path, 'Mislabelled')


    for label_dir in os.walk(labelled_path):
        os.chdir(label_dir[0])
        if glob.glob("*.*"):
            ir.resize(label_dir[0])
            calculate_fd(label_dir[0])

    for unlabel_dir in os.walk(unlabelled_path):
        os.chdir(unlabel_dir[0])
        if glob.glob("*.*"):
            ir.resize(unlabel_dir[0])
            calculate_fd(unlabel_dir[0])
    for mislabel_dir in os.walk(mislabelled_path):
        os.chdir(mislabel_dir[0])
        if glob.glob("*.*"):
            ir.resize(mislabel_dir[0])
            calculate_fd(mislabel_dir[0])

    #Creating Bags for each model and saving into Database
    # createKMeans("CM", 40)
    # createKMeans("LBP", 40)
    createKMeans("SIFT", 70)
    exit(0)

elif task_id ==1:
    if args.label == "None" or args.unlabel == "None":
        print("Please provide proper csv names")
        exit(1)
    t1 = Task1()
    t1.create_bow("SIFT", 60, args.label, args.unlabel)
    image_classified = t1.classify_DP("SIFT", k, args.label, args.unlabel)
    vz.visualize_labelled_images(image_classified[0], 30, '', 0, image_classified[1])
    exit(0)
elif task_id ==2:
    if args.label == "None" or args.unlabel == "None":
        print("Please provide proper csv names")
        exit(1)
    result = Query_input(c, args.label, args.unlabel)
    vz.visualize_labelled_images(result[0], 0, '', c, result[1])
    exit(0)
elif task_id ==3:
    if args.label == "None":
        print("Please provide Label")
        exit(1)
    if args.list == None:
        print("Please provide images in the list")
        exit(1)

    imgage_list = args.list.rstrip().split(' ')
    print(imgage_list)
    x = PersonalizedPageRank()
    result = x.getKDominantImagesUsingPPR(c, m, imgage_list, args.label)
    vz.visualize_ppr_images(imgage_list, result[0], c, m, result[1])
    # vz.visualize_ppr_images()
    exit(0)
elif task_id == 4:
    clss_typ = args.type
    if clss_typ == "None":
        print("Please provide type of Classification")
        exit(1)
    if args.label == "None" or args.unlabel == "None":
        print("Please provide proper csv names")
        exit(1)
    if clss_typ == "PPR":
        x = PersonalizedPageRank()
        result = x.classifyUnlabelledImagesUsingPPR(5, args.label, args.unlabel)
        print(result[0])
        vz.visualize_labelled_images(result[0], 0, 'PPR Based', 0, result[1])
    elif clss_typ == "SVM":
        x = SVM()
        result = x.preprocess_SVM("bag_SIFT", 70, args.label, args.unlabel)
        vz.visualize_labelled_images(result[0], 0, 'SVM Based', 0, result[1])
    elif clss_typ == "DT":
        x = DecTree()
        result = x.preprocess_dectree("HOG", 30, args.label, args.unlabel)
        vz.visualize_labelled_images(result[0], 0, 'Decision Tree Based', 0, result[1])
    else:
        print("Not a valid Classifier")
        exit(1)
    exit(0)
elif task_id == 5:
    # if args.dir == "None":
    #     print("Please provide directory name")
    #     exit(1)
    if args.imageid == "None":
        print("Please provide ImageID")
        exit(1)
    #image = os.path.join(args.dir,args.imageid)
    csv_path = os.path.join("..", "csv", "output_pca_final.csv")
    img_df = pd.read_csv(csv_path, header=None)
    # lsh = LSH()
    result = img_ann(img_df, args.imageid, m, args.layers, k)
    vz.visualize_relevance_feedback(args.imageid, result, m, args.layers, k)

    print(result)
    exit(0)



#!/usr/bin/python
import glob
import cv2
import os
import Constants as const
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('-d', '--dir', action="store", dest="dir", help="Provide directory name", default="None")
# args = parser.parse_args()


def resize(dirs):
    os.chdir(dirs)
    for file in glob.glob("*.*"):
        filepath = os.path.join(dirs, file)
        print(filepath)
        head, tail = os.path.split(filepath)
        print(tail)
        filename, extension = tail.split('.')
        # print(filename)
        img = cv2.imread(filepath)
        h, w, d = img.shape
        print(img.shape)
        if h != 1200 or w != 1600:
            img = cv2.resize(img, (1600, 1200))
        os.remove(filepath)
        cv2.imwrite(os.path.join(const.DB14_IMG_PATH, tail), img)
        # os.chdir()
        cv2.imwrite(filename + '.jpg', img)

# resize(r"C:\Users\tyler\Desktop\testTask0")

# class re_size:
#     def resize(self, dirs):
#         os.chdir(dirs)
#         for file in glob.glob("*.*"):
#             filepath=os.path.join(dirs, file)
#             filename, extension = filepath.split('.')
#             img = cv2.imread(filepath)
#             h,w,d=img.shape
#             print(img.shape)
#             if h != 1200 or w!= 1600:
#                 img = cv2.resize(img, (1600, 1200))
#             os.remove(filepath)
#             cv2.imwrite(os.path.basename(filename)+'.jpg', img)
#
# if args.dir == "None":
#     print("Please provide directory name")
#     exit(1)
# path = args.dir
# r2 = re_size()
# for file in os.walk(path):
#     # print (root)
#     # print (dirs)
#     # print (files)
#     print(file[0])
#     r2.resize(file[0])

# if __name__ == '__main__':
#     r1=re_size()
#     r1.resize(r"C:\Users\shadab\Pictures\new")
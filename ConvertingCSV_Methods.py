#!/usr/bin/env python
# coding: utf-8

# In[85]:


#def transform_images_to_csv(path_to_image_folder):
from PIL import Image
import glob
import numpy as np 
import os, os.path, time
import sys
import csv
from csv import writer
from csv import reader
import pandas as pd
from sklearn.model_selection import train_test_split
#import cv2

#creating a image list with all images of the Dataset 
def createFileList(path,format = '.jpg'):
    images_list = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for name in f:
            if name.endswith(format):
                fullName = os.path.join(r, name)
                images_list.append(fullName)
                     
    #print(images_list)
    return images_list

#creating the csv by converting every image if the image list and appending 
# the imagename as label at the end of each array/row in the csv.
# Inserts the imagelist and manual insert the final csv name, you want to have
def create_csv(myFileList, csvName):
    with open(csvName, 'a') as f:
            #Creates the csv 
            writer = csv.writer(f)
            pixel_list = []
            #first row in csv gets description names and label
            for i in range (0, 10000):
                pixel_list.append("pixel" + str(i))
            pixel_list.append("label")
            writer.writerow(pixel_list)
            
            #converting every image and adding it to csv
            for file in myFileList:
                print(file)
                img_file = Image.open(file)
                # img_file.show()

                 #Resize image to shape 100x100 and make it greyscale
                size = 100,100 #width, heigth
                #img_grey = img_file.draft('L', size)
                img_grey= img_file.resize(size)
                img_grey = img_grey.convert('L')
        
                #print(img_grey.size)
                #print(img_grey.getdata())

                # Save Greyscale values
                value = np.asarray(img_grey.getdata(), dtype=np.int)
                value = value.flatten()
                print(value.shape)
                #value = value.reshape(-1,1).T

                #concatenate the image greyscale values with the imagename to one array
                np_ar1 = value
                np_ar2 = np.array(file)

                df1 = pd.DataFrame({'ar1':np_ar1})
                df2 = pd.DataFrame({'ar2':np_ar2}, index = [0])
                new_value = pd.concat([df1.ar1, df2.ar2.T], axis=0)
                #print(new_value.shape)
                final_value = new_value.to_numpy()
                #final_value = conv_value.reshape(-1,1).T
#                 print(final_value.shape)#row,column
#                 print("----")
                
                writer.writerow(final_value)
                
def train_test_split_csv(csv,final_test_csv):
    data = pd.read_csv(csv)
    #print(data.head())
    y = data.Path
    x = data.drop('Path', axis = 1)
#     print(x.head())
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.02)
    #print(x_test.head())
    #print(x_test.shape)
    test_set = x_test
    test_set = pd.concat([y_test,x_test], axis=1)
    test_set.sort_values(by=['name'])
    print(test_set.head(5))
    print(test_set.shape)
    
    test_set.to_csv(final_test_csv, index = False, header=True)
    
#Sample Application           
# list = createFileList('yalefaces')
# create_csv(list,"yalefaces_fullDataset.csv")
# d0 = pd.read_csv('yalefaces_fullDataset.csv')
# print(d0.head(5))
# print(d0.shape)
# # train_test_split_csv('yalefaces_fullDataset.csv', 'yalefaces_testDataset.csv')


# In[ ]:


#alternative way, without images: def transform_images_to_csv(path_to_image_folder):
import pandas as pd

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

#convert("MNIST_fashion/train-images-idx3-ubyte", "MNIST_fashion/train-labels-idx1-ubyte","mnist_train.csv", 60000)

# helperMethod 
# def loadImage(path):
#     im = Image.open(path)
#     im_arr = np.asarray(im)
#     y = len(im_arr)
#     x = len(im_arr[0])
#     return x, y, im_arr 


# In[ ]:





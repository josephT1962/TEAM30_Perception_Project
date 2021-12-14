import cv2
import torch
import tqdm
import os
import numpy as np
from tqdm import tqdm
from scipy.misc import imread, imsave, imresize
from glob import glob
from torchvision import transforms

import data.rob535_task1.ref as ds
from data.rob535_task1.dp import Dataset
import train as net


def preprocess(img):
    img = Dataset.preprocess(img)
    return img[None, :, :, :]


def generate():
    """
    Generate label on the final test set
    """
    func, config = net.init()
    files = glob(ds.data_dir + '/test/*/*_image.jpg')
    print("find {} test images".format(len(files)))
    #results = {}
    f = open('result.csv', 'w')
    f.write('guid/image,label\n')

    for filename in tqdm(files):
        img = imread(filename)
        img = preprocess(img)
        output = func(-1, config, phase='inference', imgs=img)
        pred = output['preds'][0]
        pred = np.argmax(pred)

        dirs = filename.split('/')
        fdir = dirs[-2]
        fname = dirs[-1][0:4]
        fname = fdir + '/' + fname
        #results[fname] = pred
        line = fname + ',' + str(pred)
        f.write(line + '\n')
        #print(line)

    f.close()



if __name__=='__main__':
    test_set = range(0,1500)
    valid_set = range(3000, 4000)
    generate()


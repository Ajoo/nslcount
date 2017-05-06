# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os

import numpy as np
import pandas as pd

from scipy import ndimage as nd

import matplotlib.pyplot as plt

import shapely
import shapely.geometry
from shapely.geometry import Polygon


def center_of_slice(slc):
    return (slc.start+slc.stop)//2

circle7 = [[0,0,1,1,1,0,0],
           [0,1,1,1,1,1,0],
           [1,1,1,1,1,1,1],
           [1,1,1,1,1,1,1],
           [1,1,1,1,1,1,1],
           [0,1,1,1,1,1,0],
           [0,0,1,1,1,0,0]]
circle7 = np.array(circle7)
def find_circles(img):
    img = nd.morphology.binary_closing(img, circle7)
    lbl, count = nd.measurements.label(img, \
                   nd.morphology.generate_binary_structure(2,1))
    slices = nd.measurements.find_objects(lbl, count)
    centers = np.array([(center_of_slice(sl[0]), center_of_slice(sl[1])) for sl in slices])
    return centers

def merge_centers(coords, cls, tol=7):
    connect = np.sqrt(np.sum((coords-coords[:,None,:])**2, axis=-1)) < tol
    for (i, c) in enumerate(connect):
        pass
    
#def index_image(img, rclist):
#    
#    array[rc[0], rc[1], :] for rc in rclist
        
    
class SeaLionData(object):
    
    
    def __init__(self):
        self.IMG_FILE = os.path.join('..', '{itype}', '{iid}.jpg')
        self.COORDS_FILE = os.path.join('..', 'Coords', '{iid}.csv')
        self.TRAIN_COUNTS = os.path.join('..', 'Train', 'train.csv')
        
        self.n_cls = 5
        
        self.cls_names = (
            'adult_males',
            'subadult_males',
            'adult_females',
            'juveniles',
            'pups')
        self.cls_names = cls_names = pd.Series(np.arange(len(self.cls_names)), self.cls_names)
    
        # backported from @bitsofbits. Average actual color of dot centers.
        self.cls_colors = (
            (243,8,5),          # red
            (244,8,242),        # magenta
            (87,46,10),         # brown 
            (25,56,176),        # blue
            (38,174,21),        # green
            )
        self.cls_colors = np.array(self.cls_colors)
            
        self.dot_radius = 3
        
        self.n_train = 947
        self.n_test = 18636
       

        self.bad_tids = (
            # From MismatchedTrainImages.txt
            3, 
            7,    # TrainDotted rotated 180 degrees. Apply custom fix in load_dotted_image()
            9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 
            268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499, 
            507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 
            779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 
            913, 927, 946,
            # Additional
            857,    # Many sea lions, but no dots on sea lions
            )
            
        self.counts = pd.read_csv(self.TRAIN_COUNTS, index_col='train_id')

        
    def train_ids(self, start=0, end=np.inf):
        """List of all valid train ids"""
        return set(range(max(0, start), min(self.n_train, end))) - set(self.bad_tids)
         
    def test_ids(self, start=0, end=np.inf):
        """List of all valid train ids"""
        return set(range(max(0, start), min(self.n_test, end))) 

        
    def load_image(self, iid, itype='TrainDotted'):
        fname = self.IMG_FILE.format(itype=itype, iid=iid) 
        #return np.asarray(Image.open(fname))
        return nd.imread(fname)
        
    def load_mask(self, iid):
        img = self.load_image(iid, 'TrainDotted')
        return np.all(img == 0, axis=-1)
        
    def load_masked_image(self, iid):
        img = self.load_image(iid, 'Train')
        mask = self.load_mask(iid)
        img[mask, :] = 0
        return img
        
    def load_diff(self, iid):
        ZERO_TOL = 3
        MIN_DIFFERENCE = 50

        img = self.load_image(iid, 'Train').astype(np.int16)
        imgd = self.load_image(iid, 'TrainDotted').astype(np.int16)
        
        mask = np.any(imgd > ZERO_TOL, axis=-1)
        mask = nd.morphology.binary_erosion(mask, np.ones((7,7), dtype=np.bool))
        
        diff = np.max(np.abs(img-imgd), axis=-1) > MIN_DIFFERENCE
        return diff & mask
        
    def debug_coords(self, iid):
        imgd = self.load_image(iid, 'TrainDotted')
        coords = self.find_coords(iid)
        plt.imshow(imgd)
        plt.scatter(coords['x'].values, coords['y'].values, c='c', marker='+', s=40)
        
    def _find_coords(self, iid):
        ZERO_TOL = 3
        MIN_DIFFERENCE = 50
        COLOR_TOL = 200

        img = self.load_image(iid, 'Train').astype(np.int16)
        imgd = self.load_image(iid, 'TrainDotted').astype(np.int16)
        
        mask = np.any(imgd > ZERO_TOL, axis=-1)
        mask = nd.morphology.binary_erosion(mask, np.ones((10,10), dtype=np.bool))
        
        diff = np.max(np.abs(img-imgd), axis=-1) > MIN_DIFFERENCE
        centers = find_circles(diff & mask)
        
        center_colors = imgd[centers[:,0], centers[:,1]]
        cd = center_colors-sld.cls_colors[:,None,:]
        cd = np.sum(np.abs(cd), axis=-1)
        cls = np.argmin(cd, axis=0)
        cmin = np.amin(cd, axis=0)
        
        valid = cmin < COLOR_TOL
        if np.sum(~valid) > 0:
            for i in np.nonzero(~valid)[0]:
                print("Invalid colored dot detected in image {iid} at coords {coords}"\
                      .format(iid=iid, coords=centers[i]))
        return centers[valid,:], cls[valid]

    def find_coords(self, iid):
        coords, cls = self._find_coords(iid)
        coords = np.hstack((cls[:,None], coords))
        coords = pd.DataFrame(coords, columns=['class', 'y', 'x'])
        
        self.check_counts(iid, coords['class'])
        return coords
        
#    @property
#    def counts(self):
#        return pd.read_csv('../Train/train.csv', index_col='train_id')
        
    def check_counts(self, iid, cls):
        counts = self.counts.loc[iid]
        cls_counts = self.cls_names.map(cls.value_counts()).fillna(0).astype(np.int64)
        diff = cls_counts-counts
        if diff.any():
            print('Count mismatch at image {iid}'.format(iid=iid))
            print(diff)
        

    def save_coords(self, train_ids=None): 
        if train_ids is None: train_ids = self.train_ids()
        
        for iid in train_ids:
            print('Finding coordinates in image {iid}'.format(iid=iid))
            coords = self.find_coords(iid).to_csv(self.COORDS_FILE.format(iid=iid), index=False)
            
    def load_coords(self, iid):
        return pd.read_csv(self.COORDS_FILE.format(iid=iid))

            
        



if __name__ == "__main__":
    # Build coordinates
    sld = SeaLionData()
#    sld.save_coords()
#
#    # Error analysis
#    sld.verbosity = VERBOSITY.VERBOSE
#    tid_counts = sld.count_coords(sld.tid_coords)
#    rmse, frac = sld.rmse(tid_counts)
#
#    print()
#    print('RMSE: {}'.format(rmse) )
#print('Error frac: {}'.format(frac))
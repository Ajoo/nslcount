# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import os
import globs

import numpy as np
import pandas as pd

from scipy import ndimage as nd
from scipy.misc import imsave
import cv2


def center_of_slice(slc):
    return (slc.start+slc.stop)//2

def find_circles(img, median_k=3):
    params = cv2.SimpleBlobDetector_Params()

    params.blobColor = 255
    params.minDistBetweenBlobs = 3 
    #         Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 100
    params.thresholdStep = 3
     
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
     
    # Filter by Area.
    params.filterByArea = True
    params.maxArea = 100
    params.minArea = 3
             
    detector = cv2.SimpleBlobDetector_create(params)
            
    keypoints = detector.detect(img)
    coords = np.array(list(map(lambda kp: kp.pt, keypoints)))[:,::-1]
    sizes = np.array(list(map(lambda kp: kp.size, keypoints)))
    return coords, sizes

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
        self.TILE_FILE = os.path.join('..', 'Tiles', '{itype}', '{cls}', '{iid}_{nid}.jpg')
        self.TRAIN_COUNTS = os.path.join('..', 'Train', 'train.csv')

        
        self.TILE_SIZE = 64
        
        self.n_cls = 5
        
        self.cls_names = (
            'adult_males',
            'subadult_males',
            'adult_females',
            'juveniles',
            'pups',
            'not_sea_lion')
        self.cls_ids = pd.Series(np.arange(len(self.cls_names)-1), self.cls_names[:-1])
    
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
        
    def tile_folder(itype='Train'):
        return os.path.join('..', 'Tiles', itype)
        
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

        img = self.load_image(iid, 'Train')
        imgd = self.load_image(iid, 'TrainDotted')
        
        mask = np.all(imgd < ZERO_TOL, axis=-1)
        mask = nd.morphology.binary_dilation(mask, np.ones((7,7), dtype=np.bool))
        
        diff = cv2.absdiff(img, imgd)
        diff = np.max(diff, axis=-1)
        diff[mask] = 0
        return diff, imgd
        
    def debug_coords(self, iid):
        imgd = self.load_image(iid, 'TrainDotted')
        coords = self.find_coords(iid)
        plt.imshow(imgd)
        plt.scatter(coords['x'].values, coords['y'].values, c='c', marker='+', s=40)
        
        
    def _find_coords(self, iid, median_k=5):
        COLOR_TOL = 200

        diff, imgd = self.load_diff(iid)
        diff = cv2.medianBlur(diff, median_k)
        centers, sizes = find_circles(diff)
        centers = np.round(centers).astype(np.int)

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
        coords = pd.DataFrame(coords, columns=['class', 'col', 'row'])
        
        self.check_counts(iid, coords['class'])
        return coords
        
#    @property
#    def counts(self):
#        return pd.read_csv('../Train/train.csv', index_col='train_id')
        
    def check_counts(self, iid, cls):
        counts = self.counts.loc[iid]
        cls_counts = self.cls_ids.map(cls.value_counts()).fillna(0).astype(np.int64)
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

    def convert_coords(self, train_ids=None):
        if train_ids is None: train_ids = self.train_ids()
        
        coords = pd.read_csv('../gavin_coords.csv')
        coords.columns = ['tid', 'class', 'row', 'col']
        for tid in self.train_ids():
            tid_df = coords[coords['tid']==tid].loc[:,('class', 'row', 'col')]
            tid_df.to_csv(self.COORDS_FILE.format(iid=tid), index=False)
        

    def purge_tiles(self):
        files = glob.glob('../Tiles/*/*/*.jpg')
        for fname in files:
            os.remove(fname)
            
    def extract_tiles(self, iids, ttype='Train'):
        
        for iid in iids:
            coords = self.load_coords(iid)
            img = self.load_image(iid, 'Train')
            
            for nid, (tile, cls) in enumerate(self._extract_sea_lions(img, coords)):
                fname = self.TILE_FILE.format(cls=self.cls_names[cls], itype=ttype, iid=iid, nid=nid)
                imsave(fname, tile)
                
            for nid, tile in enumerate(self._extract_empty_tiles(img, coords)):
                fname = self.TILE_FILE.format(cls='not_sea_lion', itype=ttype, iid=iid, nid=nid)
                imsave(fname, tile)
    
    def extract_sea_lions(self, iid, ttype='Train'):
        coords = self.load_coords(iid)
        img = self.load_image(iid, 'Train')
        
        for nid, (tile, cls) in enumerate(self._extract_sea_lions(img, coords)):
            fname = self.TILE_FILE.format(cls=self.cls_names[cls], itype=ttype, iid=iid, nid=nid)
            imsave(fname, tile)
            
    def extract_empty_tiles(self, iid, ttype='Train'):
        coords = self.load_coords(iid)
        img = self.load_image(iid, 'Train')
        
        for nid, tile in enumerate(self._extract_empty_tiles(img, coords)):
            fname = self.TILE_FILE.format(cls='not_sea_lion', itype=ttype, iid=iid, nid=nid)
            imsave(fname, tile)
                        
            
    def _extract_sea_lions(self, img, coords):
        h = self.TILE_SIZE//2

        for cls, x, y in coords.loc[:,('class', 'col', 'row')].values:
            if x < h or x > img.shape[1] - h or y < h or y > img.shape[0] - h:
                continue
            yield img[y-h:y+h, x-h:x+h, :], cls
            
            
    def _extract_empty_tiles(self, img, coords, ttype='Train'):
        N_SAMPLES = 50
        h = self.TILE_SIZE//2
        
        y = np.random.randint(h, img.shape[0]-h, N_SAMPLES)
        x = np.random.randint(h, img.shape[1]-h, N_SAMPLES)
        rcoords = np.hstack((y[:,None], x[:,None]))
        coords = coords.loc[:,('row', 'col')].values
        
        d = np.sum(np.abs(rcoords[:,None,:]-coords), axis=-1)
        dmin = np.min(d, axis=-1)
        
        rcoords = rcoords[dmin > h]
        for y, x in rcoords:
            yield img[y-h:y+h, x-h:x+h, :]

        

if __name__ == "__main__":
    sld = SeaLionData()
    
    sld.purge_tiles()
    
    train_ids = sld.train_ids(0, 100)
    val_ids = sld.train_ids(100, 150)
    test_ids = sld.train_ids(150, 200)

    sld.extract_tiles(train_ids, 'Train')
    sld.extract_tiles(val_ids, 'Val')
    sld.extract_tiles(test_ids, 'Test')
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 15:04:56 2017

@author: ajoo
"""
from skimage.feature import blob_log

iid = 1
diff, imgd = sld.load_diff(iid)
cdiff = imgd.astype(np.float)-sld.cls_colors[:,None,None,:]
cd = np.sqrt(np.amin(np.sum(np.abs(cdiff), axis=-1), axis=0))
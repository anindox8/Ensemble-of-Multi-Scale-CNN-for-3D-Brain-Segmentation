from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import SimpleITK as sitk
import os
import numpy as np
from dltk.io.augmentation import add_gaussian_noise, flip, extract_class_balanced_example_array
import scipy.ndimage
import time

 
'''
 Multi-Class Segmentation (3D)
 Feed: Data Generator

 Update: 30/12/2019
 Contributors: anindox8
 
 // Target Organ (1):     
     - Brain

 // Classes (3):             
     - Cerebrospinal Fluid (CS)
     - Gray Matter (GM)
     - White Matter (WM)
'''

def read_fn(file_references, mode, params=None):
    """A custom python read function for interfacing with nii image files.
    Args:
        file_references (list):
        mode (str): One of the tf.estimator.ModeKeys strings: TRAIN, EVAL or
            PREDICT.
        params (dict, optional): A dictionary to parametrise read_fn outputs
            (e.g. reader_params = {'n_patches': 10, 'patch_size':
            [64, 64, 64], 'extract_patches': True}, etc.).
    Yields:
        dict: A dictionary of reader outputs for dltk.io.abstract_reader.
    """

    # Data Augmentation (Gaussian Blurring, Axial-Plane Horizontal Flip)
    def _augment(img,lbl):
        if (np.random.randint(0,10)<3):
          img      = scipy.ndimage.gaussian_filter(img, sigma=0.25)
        [img, lbl] = flip([img, lbl], axis=0)
        return img, lbl

    # Crop Central Block Matching Output of DeepMedic
    def crop_central_block_label(img,cropz,cropy,cropx):
        _,z,y,x = img.shape
        startx  = x//2-(cropx//2)-1
        starty  = y//2-(cropy//2)-1
        startz  = z//2-(cropz//2)-1        
        return img[:,startz:startz+cropz,starty:starty+cropy,startx:startx+cropx]

    for f in file_references:
        t0 = time.time()  

        scan_id    = str(f[0])
        img_itk    = sitk.ReadImage(str(f[1]), sitk.sitkFloat32)
        img        = np.expand_dims(np.array(sitk.GetArrayFromImage(img_itk)), axis=3)
        lbl        = np.array(sitk.GetArrayFromImage(sitk.ReadImage(str(f[2])))).astype(np.int32)

        print('Loaded {}; Time = {}'.format(scan_id,(time.time()-t0)))

        # Testing Mode
        if (mode == tf.estimator.ModeKeys.PREDICT):
            yield {'features':  {'x': img},
                   'labels':      None,
                   'sitk':        img_itk,
                   'subject_id':  scan_id}
    
        # Training Mode
        if (mode == tf.estimator.ModeKeys.TRAIN):
            img, lbl = _augment(img, lbl) 

        # Return Training Examples
        if params['extract_patches']:
            img, lbl = extract_class_balanced_example_array(img, lbl,
                               example_size  = params['patch_size'],
                               n_examples    = params['n_patches'],
                               classes       = 4,
                               class_weights = [0,1,1,1])

            lbl = crop_central_block_label(lbl,9,9,9)

            for e in range(params['n_patches']):
                yield {'features': {'x': img[e].astype(np.float32)},
                       'labels':   {'y': lbl[e].astype(np.int32)},
                       'img_id':         scan_id}

        # Return Full Images
        else:
            yield {'features': {'x': img},
                   'labels':   {'y': lbl},
                   'sitk':       img_itk,
                   'img_id':     scan_id}

    return

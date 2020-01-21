import os
import numpy as np
import operator
import SimpleITK as sitk
import pandas as pd
from dltk.io.preprocessing import whitening

'''
 Multi-Class Segmentation (3D)
 Preprocess: Resample Resolution + Normalization

 Update: 30/12/2019
 Contributors: anindox8
 
 // Target Organ (1):     
     - Brain

 // Classes (3):             
     - Cerebrospinal Fluid (CS)
     - Gray Matter (GM)
     - White Matter (WM)
'''

# Resample Images to 1mm Spacing with SimpleITK
def resample_img1mm(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):

    original_spacing = itk_image.GetSpacing()
    original_size    = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


# Master Directories
raw_data_path       = '/home/cad1/anindo/neuro/data/original/'
raw_save_path       = '/home/cad1/anindo/neuro/data/preprocess/'


# Setting I/O Directories
train_data_list     = os.listdir(raw_data_path + 'train_images/')
val_data_list       = os.listdir(raw_data_path + 'val_images/')
test_data_list      = os.listdir(raw_data_path + 'test_images/')


# Preprocess Training Images
print('Preprocess Training Images...')
for i in range(len(train_data_list)):
    original_image  = sitk.ReadImage(raw_data_path + 'train_images/' + train_data_list[i], sitk.sitkFloat32)
    original_label  = sitk.ReadImage(raw_data_path + 'train_labels/' + train_data_list[i].replace('.nii.gz','_seg.nii.gz'))
    
    # Resample to 1mm Uniform Cubic Resolution
    resampled_image = resample_img1mm(original_image)
    resampled_label = resample_img1mm(original_label,is_label=True)

    # Whitening (Mean=0, Std=1)
    whitened_image  = sitk.GetImageFromArray(whitening(np.array(sitk.GetArrayFromImage(resampled_image))))

    # Store Preprocessed Images
    sitk.WriteImage(resampled_image, str(raw_save_path + 'train_images/'+ train_data_list[i]))
    sitk.WriteImage(resampled_label, str(raw_save_path + 'train_labels/'+ train_data_list[i].replace('.nii.gz','_seg.nii.gz')))
    print('Preprocessed:{}'.format(train_data_list[i]))
print('Complete.')

# Preprocess Validation Images
print('Preprocess Validation Images...')
for i in range(len(val_data_list)):
    original_image  = sitk.ReadImage(raw_data_path + 'val_images/' + val_data_list[i], sitk.sitkFloat32)
    original_label  = sitk.ReadImage(raw_data_path + 'val_labels/' + val_data_list[i].replace('.nii.gz','_seg.nii.gz'))
    
    # Resample to 1mm Uniform Cubic Resolution
    resampled_image = resample_img1mm(original_image)
    resampled_label = resample_img1mm(original_label,is_label=True)

    # Whitening (Mean=0, Std=1)
    whitened_image  = sitk.GetImageFromArray(whitening(np.array(sitk.GetArrayFromImage(resampled_image))))

    # Store Preprocessed Images
    sitk.WriteImage(resampled_image, str(raw_save_path + 'val_images/'+ val_data_list[i]))
    sitk.WriteImage(resampled_label, str(raw_save_path + 'val_labels/'+ val_data_list[i].replace('.nii.gz','_seg.nii.gz')))
    print('Preprocessed:{}'.format(val_data_list[i]))
print('Complete.')

# Preprocess Testing Images
print('Preprocess Testing Images...')
for i in range(len(test_data_list)):
    original_image  = sitk.ReadImage(raw_data_path + 'test_images/' + test_data_list[i], sitk.sitkFloat32)
    
    # Resample to 1mm Uniform Cubic Resolution
    resampled_image = resample_img1mm(original_image)

    # Whitening (Mean=0, Std=1)
    whitened_image  = sitk.GetImageFromArray(whitening(np.array(sitk.GetArrayFromImage(resampled_image))))

    # Store Preprocessed Images
    sitk.WriteImage(resampled_image, str(raw_save_path + 'test_images/'+ test_data_list[i]))
    print('Preprocessed:{}'.format(test_data_list[i]))
print('Complete.')
import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold


'''
 Multi-Class Segmentation (3D)
 Feed: Generating CSV for I/O

 Update: 30/12/2019
 Contributors: anindox8
 
 // Target Organ (1):     
     - Brain

 // Classes (3):             
     - Cerebrospinal Fluid (CS)
     - Gray Matter (GM)
     - White Matter (WM)
'''


# Feeding Script Input Parameters
raw_data_path       = '/home/cad1/anindo/neuro/data/preprocess/'
save_path           = '/home/cad1/anindo/neuro/feed/'
val_mode            = 'single'     
folds               =  10

# Setting I/O Directories
train_data_list     = os.listdir(raw_data_path + 'train_images/')
val_data_list       = os.listdir(raw_data_path + 'val_images/')
test_data_list      = os.listdir(raw_data_path + 'test_images/')

if (val_mode=='single'):
    print('Generating CSV Feeder for Single-Fold Validation...')
    # Populating Lists from Training Data
    scan_id_list        = []
    image_path_list     = []
    label_path_list     = []

    for i in range(len(train_data_list)):
        scan_id_list.append(train_data_list[i].split('.nii.gz')[0])
        image_path_list.append(raw_data_path + 'train_images/' + train_data_list[i])
        label_path_list.append(raw_data_path + 'train_labels/' + train_data_list[i].replace('.nii.gz','_seg.nii.gz'))

    # Synchronous Data Shuffle
    a_train, b_train, c_train  = shuffle(scan_id_list, image_path_list, label_path_list, random_state=8)

    # Metadata Setup
    trainData                  = pd.DataFrame(list(zip(a_train,b_train,c_train)),
    columns                    = ['scan_id', 'image_path', 'label_path'])
    trainData_name             = save_path + 'Training-Fold.csv'
    trainData.to_csv(trainData_name, encoding='utf-8', index=False)

    # Populating Lists from Validation Data
    scan_id_list        = []
    image_path_list     = []
    label_path_list     = []

    for i in range(len(val_data_list)):
        scan_id_list.append(val_data_list[i].split('.nii.gz')[0])
        image_path_list.append(raw_data_path + 'val_images/' + val_data_list[i])
        label_path_list.append(raw_data_path + 'val_labels/' + val_data_list[i].replace('.nii.gz','_seg.nii.gz'))

    # Synchronous Data Shuffle
    a_val, b_val, c_val  = shuffle(scan_id_list, image_path_list, label_path_list, random_state=8)

    # Metadata Setup
    valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val)),
    columns                    = ['scan_id', 'image_path', 'label_path'])
    valData_name               = save_path + 'Validation-Fold.csv'
    valData.to_csv(valData_name, encoding='utf-8', index=False)
    print('Complete.')


elif (val_mode=='kfold'):
    print('Generating CSV Feeder for {}-Fold Validation...'.format(str(folds)))
    # Populating Lists from Training Data
    for i in range(len(train_data_list)):
        scan_id_list.append(train_data_list[i].split('.nii.gz')[0])
        image_path_list.append(raw_data_path + 'train_images/' + train_data_list[i])
        label_path_list.append(raw_data_path + 'train_labels/' + train_data_list[i].replace('.nii.gz','_seg.nii.gz'))

    # Synchronous Data Shuffle
    a, b, c  = shuffle(scan_id_list, image_path_list, label_path_list, random_state=8)

    # Metadata Setup
    shuffled_dataset  = pd.DataFrame(list(zip(a,b,c)),
    columns           = ['scan_id', 'image_path','label_path'])
    scan_id           = shuffled_dataset['scan_id']
    image_path        = shuffled_dataset['image_path']
    label_path        = shuffled_dataset['label_path']

    # Generating CSV
    kf     = KFold(folds)
    fold   = 0
    for train, val in kf.split(a):
        fold +=1
        print('Fold #' + str(fold))

        a_train = scan_id[train]
        b_train = image_path[train]
        c_train = label_path[train]
        a_val   = scan_id[val]
        b_val   = image_path[val]
        c_val   = label_path[val]

        trainData                  = pd.DataFrame(list(zip(a_train,b_train,c_train)),
        columns                    = ['scan_id', 'image_path', 'label_path'])
        trainData_name             = save_path + 'Training-Fold-{}'.format(fold)+'.csv'
        trainData.to_csv(trainData_name, encoding='utf-8', index=False)

        valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val)),
        columns                    = ['scan_id', 'image_path', 'label_path'])
        valData_name               = save_path + 'Validation-Fold-{}'.format(fold)+'.csv'
        valData.to_csv(valData_name, encoding='utf-8', index=False)
    print('Complete.')






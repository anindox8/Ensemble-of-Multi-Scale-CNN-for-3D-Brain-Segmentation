from __future__ import division
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk
from tensorflow.contrib import predictor
from dltk.core import metrics as metrics
from dltk.utils import sliding_window_segmentation_inference
from data_reader import read_fn
import warnings


'''
 Multi-Class Segmentation (3D)
 Deploy: Predicting Segmentation Volumes

 Update: 06/01/2020
 Contributors: anindox8
 
 // Target Organ (1):     
     - Brain

 // Classes (3):             
     - Cerebrospinal Fluid (CS)
     - Gray Matter (GM)
     - White Matter (WM)
'''


READER_PARAMS         = {'extract_patches': False}
BATCH_SIZE            = 1
ENSEMBLE_MODE         = 'weighted_mean'
EXECUTION_MODE        = 'VAL'

def predict(args):
    
    # Read List of Validation/Testing Samples
    file_names = pd.read_csv(
        args.csv,
        dtype=object,
        keep_default_na=False,
        na_values=[]).values

    # Load Pre-Trained Model
    export_dir01 = [os.path.join(args.model01_path, o) for o in os.listdir(args.model01_path)
                   if os.path.isdir(os.path.join(args.model01_path, o)) and o.isdigit()][-1]    
    export_dir02 = [os.path.join(args.model02_path, o) for o in os.listdir(args.model02_path)
                   if os.path.isdir(os.path.join(args.model02_path, o)) and o.isdigit()][-1]
    export_dir03 = [os.path.join(args.model03_path, o) for o in os.listdir(args.model03_path)
                   if os.path.isdir(os.path.join(args.model03_path, o)) and o.isdigit()][-1]
    export_dir04 = [os.path.join(args.model04_path, o) for o in os.listdir(args.model04_path)
                   if os.path.isdir(os.path.join(args.model04_path, o)) and o.isdigit()][-1]

    predictor01  = predictor.from_saved_model(export_dir01)
    predictor02  = predictor.from_saved_model(export_dir02)    
    predictor03  = predictor.from_saved_model(export_dir03)
    predictor04  = predictor.from_saved_model(export_dir04)

    print('Pre-Trained Models Loaded.')

    # Fetch Output Probability of Trained Network
    y_prob01    = predictor01._fetch_tensors['y_prob']
    y_prob02    = predictor02._fetch_tensors['y_prob']
    y_prob03    = predictor03._fetch_tensors['y_prob']
    y_prob04    = predictor04._fetch_tensors['y_prob']

    num_classes = y_prob01.get_shape().as_list()[-1]

    if (EXECUTION_MODE=='TEST'):
        KEY = tf.estimator.ModeKeys.PREDICT
    elif (EXECUTION_MODE=='VAL'):
        KEY = tf.estimator.ModeKeys.EVAL

    # Iterate through Files, Predict on Full Volumes and Compute Dice Coefficient
    for output in read_fn(file_references = file_names,
                          mode            = KEY,
                          params          = READER_PARAMS):
        t0 = time.time()

        img = np.expand_dims(output['features']['x'], axis=0)

        # Sliding Window Inference with DLTK Wrapper
        pred01 = sliding_window_segmentation_inference(
            session=predictor01.session,
            ops_list=[y_prob01],
            sample_dict={predictor01._feed_tensors['x']: img},
            batch_size=BATCH_SIZE)[0]
        pred02 = sliding_window_segmentation_inference(
            session=predictor02.session,
            ops_list=[y_prob02],
            sample_dict={predictor02._feed_tensors['x']: img},
            batch_size=BATCH_SIZE)[0]
        pred03 = sliding_window_segmentation_inference(
            session=predictor03.session,
            ops_list=[y_prob03],
            sample_dict={predictor03._feed_tensors['x']: img},
            batch_size=BATCH_SIZE)[0]
        pred04 = sliding_window_segmentation_inference(
            session=predictor04.session,
            ops_list=[y_prob04],
            sample_dict={predictor04._feed_tensors['x']: img},
            batch_size=BATCH_SIZE)[0]

        # Calculate Prediction from Probabilities
        if (ENSEMBLE_MODE=='weighted_mean'):
            pred = (0.20*pred01 + 0.35*pred02 + 0.30*pred03 + 0.15*pred04)
        elif (ENSEMBLE_MODE=='maxconf'):
            pred = np.maximum(np.maximum(pred01,pred02),np.maximum(pred03,pred04))
        
        pred = np.argmax(pred,-1)

        # Save Ensemble Prediction
        output_fn = os.path.join(str(args.output_path + 'ensemble/'), '{}_seg.nii.gz'.format(output['img_id']))
        new_sitk  = sitk.GetImageFromArray(pred[0].astype(np.int32))
        new_sitk.CopyInformation(output['sitk'])
        sitk.WriteImage(new_sitk, output_fn)

        # Save Member Predictions
        pred01_op = np.argmax(pred01,-1)
        output_fn = os.path.join(str(args.output_path + 'resfcn32-350E/'), '{}_seg.nii.gz'.format(output['img_id']))
        new_sitk  = sitk.GetImageFromArray(pred01_op[0].astype(np.int32))
        new_sitk.CopyInformation(output['sitk'])
        sitk.WriteImage(new_sitk, output_fn)
        
        pred02_op = np.argmax(pred02,-1)
        output_fn = os.path.join(str(args.output_path + 'resfcn96-350E/'), '{}_seg.nii.gz'.format(output['img_id']))
        new_sitk  = sitk.GetImageFromArray(pred02_op[0].astype(np.int32))
        new_sitk.CopyInformation(output['sitk'])
        sitk.WriteImage(new_sitk, output_fn)

        pred03_op = np.argmax(pred03,-1)
        output_fn = os.path.join(str(args.output_path + 'resfcn112-350E/'), '{}_seg.nii.gz'.format(output['img_id']))
        new_sitk  = sitk.GetImageFromArray(pred03_op[0].astype(np.int32))
        new_sitk.CopyInformation(output['sitk'])
        sitk.WriteImage(new_sitk, output_fn)

        pred04_op = np.argmax(pred04,-1)
        output_fn = os.path.join(str(args.output_path + 'resunet112-250E/'), '{}_seg.nii.gz'.format(output['img_id']))
        new_sitk  = sitk.GetImageFromArray(pred04_op[0].astype(np.int32))
        new_sitk.CopyInformation(output['sitk'])
        sitk.WriteImage(new_sitk, output_fn)

        if (EXECUTION_MODE=='VAL'):
            # Calculate Dice Coefficient
            lbl  = np.expand_dims(output['labels']['y'], axis=0)
            dsc  = metrics.dice(pred, lbl, num_classes)[1:]
            dsc1 = metrics.dice(pred01_op, lbl, num_classes)[1:]
            dsc2 = metrics.dice(pred02_op, lbl, num_classes)[1:]
            dsc3 = metrics.dice(pred03_op, lbl, num_classes)[1:]
            dsc4 = metrics.dice(pred04_op, lbl, num_classes)[1:]
            avd  = metrics.abs_vol_difference(pred, lbl, num_classes)[1:]
            avd1 = metrics.abs_vol_difference(pred01_op, lbl, num_classes)[1:]
            avd2 = metrics.abs_vol_difference(pred02_op, lbl, num_classes)[1:]
            avd3 = metrics.abs_vol_difference(pred03_op, lbl, num_classes)[1:]
            avd4 = metrics.abs_vol_difference(pred04_op, lbl, num_classes)[1:]
            print('ID='+str(output['img_id']))
            print('Dice Score:')
            print(dsc1)
            print(dsc2)
            print(dsc3)
            print(dsc4)
            print(dsc)
            print('Absolute Volume Difference:')
            print(avd1)
            print(avd2)
            print(avd3)
            print(avd4)
            print(avd)


if __name__ == '__main__':

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Argument Parser Setup
    parser = argparse.ArgumentParser(description='Multi-Class Brain Segmentation')
    parser.add_argument('--verbose',            default=False, action='store_true')

    parser.add_argument('--cuda_devices', '-c',  default='0')
    
    if (EXECUTION_MODE=='VAL'):
        parser.add_argument('--model01_path', '-p1', default='/home/cad1/anindo/neuro/model_weights/train-val/ensemble/resfcn32-350E/Step2399ValLoss0.35591/')
        #parser.add_argument('--model01_path', '-p1', default='/home/cad1/anindo/neuro/model_weights/train-val/ensemble/resfcn64-350E/Step4356ValLoss0.22311/')
        parser.add_argument('--model02_path', '-p2', default='/home/cad1/anindo/neuro/model_weights/train-val/ensemble/resfcn96-315E/Step7821ValLoss0.16162/')
        parser.add_argument('--model03_path', '-p3', default='/home/cad1/anindo/neuro/model_weights/train-val/ensemble/resfcn112-350E/Step8624ValLoss0.13703/')
        parser.add_argument('--model04_path', '-p4', default='/home/cad1/anindo/neuro/model_weights/train-val/ensemble/resunet112-250E/Step4000ValLoss0.19733/')
        parser.add_argument('--output_path',  '-o',  default='/home/cad1/anindo/neuro/predictions/val/')
        parser.add_argument('--csv',                 default='/home/cad1/anindo/neuro/feed/Validation-Fold-O.csv')

    elif (EXECUTION_MODE=='TEST'):
        parser.add_argument('--model01_path', '-p1', default='/home/cad1/anindo/neuro/model_weights/test/ensemble/resfcn64-350E/Step4356ValLoss0.18464/')
        parser.add_argument('--model02_path', '-p2', default='/home/cad1/anindo/neuro/model_weights/test/ensemble/resfcn96-350E/Step7789ValLoss0.11406/')
        parser.add_argument('--model03_path', '-p3', default='/home/cad1/anindo/neuro/model_weights/test/ensemble/resfcn112-350E/Step8976ValLoss0.09189/')
        parser.add_argument('--model04_path', '-p4', default='/home/cad1/anindo/neuro/model_weights/test/ensemble/resunet112-250E/Step6000ValLoss0.15900/')
        parser.add_argument('--output_path',  '-o',  default='/home/cad1/anindo/neuro/predictions/test/')
        parser.add_argument('--csv',                 default='/home/cad1/anindo/neuro/feed/Testing-Fold-O.csv')

    args = parser.parse_args()

    # Set Verbosity
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU Allocation Options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Allow GPU Usage Growth
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # Deploy
    predict(args)
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
import numpy as np
import argparse
import os
import pandas as pd
import json
import warnings
from dltk.core.activations import leaky_relu
from dltk.io.abstract_reader import Reader
from dltk.core.metrics import dice
from deepmedic_reader import read_fn
from cyclic_lr import cyclic_learning_rate
from models.deepmedic import deepmedic_3d, crop_central_block
from models.unet import residual_unet_3d


'''
 Multi-Class Segmentation (3D)
 Train: Train CNN

 Update: 30/12/2019
 Contributors: anindox8
 
 // Target Organ (1):     
     - Brain

 // Classes (3):             
     - Cerebrospinal Fluid (CS)
     - Gray Matter (GM)
     - White Matter (WM)
'''


# Image Parameters
NUM_PATCHES         =   20
PATCH_XY            =   57
PATCH_Z             =   57
NUM_CLASSES         =   4
NUM_CHANNELS        =   1
TRAIN_SIZE          =   10*NUM_PATCHES


# Training Hyperparameters
MAX_EPOCHS          =   200
BATCH_SIZE          =   8
OPTIM               =  'adam'

eLR_INITIAL         =   0.001
eLRDECAY_EPOCHS     =   5     
eLRDECAY_RATE       =   0.95

CLR_STEPFACTOR      =   10
CLR_MODE            =  'exp_range'
CLR_GAMMA           =   0.9999
CLR_MINLR           =   0.0005
CLR_MAXLR           =   0.001
LR_MODE             =  'CLR'
VAL_POINTS          =   40


# Derived Operational Parameters
PREFETCH_CACHE_SIZE =   BATCH_SIZE*4
SHUFFLE_CACHE_SIZE  =   64
MAX_STEPS           =   int(np.ceil((TRAIN_SIZE/BATCH_SIZE)*MAX_EPOCHS))
EVAL_EVERY_N_STEPS  =   int(np.ceil(MAX_STEPS/VAL_POINTS))
EVAL_STEPS          =   int(np.ceil(TRAIN_SIZE/BATCH_SIZE))
eLRDECAY_STEPS      =   int(np.floor((TRAIN_SIZE/BATCH_SIZE)*eLRDECAY_EPOCHS))
CLR_STEPSIZE        =   int(np.ceil((TRAIN_SIZE/BATCH_SIZE)*CLR_STEPFACTOR))

count_steps = []
count_loss  = []

def lrelu(x):
    return leaky_relu(x, 0.1)

def model_fn(features, labels, mode, params):

    # Model Definition
    model_output_ops = deepmedic_3d(
        inputs                   = features['x'],
        num_classes              = NUM_CLASSES,
        normal_input_shape       = (25, 25, 25),
        subsampled_input_shapes  = ((PATCH_Z, PATCH_XY, PATCH_XY),),
        subsample_factors        = ((3, 3, 3),),
        mode                     = mode,
        kernel_initializer       = tf.initializers.variance_scaling(distribution='uniform'),
        bias_initializer         = tf.zeros_initializer(),
        kernel_regularizer       = tf.contrib.layers.l2_regularizer(1e-3))

    # Prediction Mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode            = mode,
            predictions     = model_output_ops,
            export_outputs  = {'out': tf.estimator.export.PredictOutput(model_output_ops)})
    
    # Loss Function
    ce   = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output_ops['logits'],
                                                          labels=labels['y'])
    loss = tf.reduce_mean(ce)
    global_step = tf.train.get_global_step()
    
    # Learning Rate
    if (LR_MODE=='eLR'):
        # Exponential Learning Rate Decay
        learning_rate = tf.train.exponential_decay(eLR_INITIAL, global_step, decay_steps=eLRDECAY_STEPS, decay_rate=eLRDECAY_RATE, staircase=True)           
    elif (LR_MODE=='CLR'):
        # Cyclic Learning Rate 
        learning_rate = cyclic_learning_rate(global_step=global_step, learning_rate=CLR_MINLR, max_lr=CLR_MAXLR, step_size=CLR_STEPSIZE, gamma=CLR_GAMMA, mode=CLR_MODE)

    # Optimizer
    if   (OPTIM == 'adam'):
        optimiser = tf.train.AdamOptimizer(
            learning_rate=learning_rate, epsilon=1e-5)
    elif (OPTIM == 'momentum'):
        optimiser = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9)
    elif (OPTIM == 'rmsprop'):
        optimiser = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate, momentum=0.9)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimiser.minimize(loss, global_step=global_step)
    
    # Custom Image Summaries (TensorBoard)
    image_summaries = {}
    image_summaries['MRI_Patch']   = features['x'][0, 27, :, :, 0]
    image_summaries['Predictions'] = tf.cast(model_output_ops['y_'], tf.float32)[0, 4, :, :]

    expected_output_size = [1, PATCH_XY, PATCH_XY, 1]  # [B, W, H, C]
    tf.summary.image('MRI_Patch',   tf.reshape(image_summaries['MRI_Patch'],   expected_output_size))
    tf.summary.image('Predictions', tf.reshape(image_summaries['Predictions'], [1, 9, 9, 1]))    

    # Track Dice Metric (TensorBoard)
    dice_tensor = tf.py_func(dice, [model_output_ops['y_'],
                                    labels['y'],
                                    tf.constant(NUM_CLASSES)], tf.float32)
    [tf.summary.scalar('DSC_CLASS_{}'.format(i), dice_tensor[i])
     for i in range(NUM_CLASSES)]
    
    # Return EstimatorSpec Object
    return tf.estimator.EstimatorSpec(mode            = mode,
                                      predictions     = model_output_ops,
                                      loss            = loss,
                                      train_op        = train_op,
                                      eval_metric_ops = None)


def train(args):
    np.random.seed(8)
    tf.set_random_seed(8)

    print('Setting Up...')

    # Read Training-Fold.csv
    train_filenames = pd.read_csv(
        args.train_csv, dtype=object, keep_default_na=False,
        na_values=[]).values

    # Read Validation-Fold.csv
    val_filenames = pd.read_csv(
        args.val_csv, dtype=object, keep_default_na=False,
        na_values=[]).values


    # Set DLTK Reader Parameters (No. of Patches, Patch Size) 
    reader_params = {'n_patches':        NUM_PATCHES,
                     'patch_size':       [PATCH_Z, PATCH_XY, PATCH_XY], # Target Patch Size
                     'label_size':       [9, 9, 9],                     # Target Patch Size
                     'extract_patches':  True}                          # Enable Training Mode Patch Extraction
    
    # Set Patch Dimensions
    reader_patch_shapes = {'features': {'x': reader_params['patch_size'] + [NUM_CHANNELS,]},
                           'labels':   {'y': reader_params['label_size']}}
    
    # Initiate Data Reader + Patch Extraction
    reader = Reader(read_fn,
                  {'features': {'x': tf.float32},
                   'labels':   {'y': tf.int32}})


    # Create Input Functions + Queue Initialisation Hooks for Training/Validation Data
    train_input_fn, train_qinit_hook = reader.get_inputs(
        file_references       = train_filenames,
        mode                  = tf.estimator.ModeKeys.TRAIN,
        example_shapes        = reader_patch_shapes,
        batch_size            = BATCH_SIZE,
        shuffle_cache_size    = SHUFFLE_CACHE_SIZE,
        prefetch_cache_size   = PREFETCH_CACHE_SIZE,
        params                = reader_params)

    val_input_fn, val_qinit_hook = reader.get_inputs(
        file_references       = val_filenames,
        mode                  = tf.estimator.ModeKeys.EVAL,
        example_shapes        = reader_patch_shapes,
        batch_size            = BATCH_SIZE,
        shuffle_cache_size    = SHUFFLE_CACHE_SIZE,
        prefetch_cache_size   = PREFETCH_CACHE_SIZE,
        params                = reader_params)


    # Instantiate Neural Network Estimator
    nn = tf.estimator.Estimator(
        model_fn             = model_fn,
        model_dir            = args.model_path,
        config               = tf.estimator.RunConfig())                                         


    # Hooks for Validation Summaries
    val_summary_hook = tf.contrib.training.SummaryAtEndHook(os.path.join(args.model_path, 'eval'))
    step_cnt_hook    = tf.train.StepCounterHook(every_n_steps = EVAL_EVERY_N_STEPS,
                                                output_dir    = args.model_path)


    print('Begin Training...')
    try:
        for _ in range(MAX_STEPS // EVAL_EVERY_N_STEPS):
            nn.train(
                input_fn  = train_input_fn,
                hooks     = [train_qinit_hook, step_cnt_hook],
                steps     = EVAL_EVERY_N_STEPS)

            if args.run_validation:
                results_val   = nn.evaluate(
                    input_fn  = val_input_fn,
                    hooks     = [val_qinit_hook, val_summary_hook],
                    steps     = EVAL_STEPS)
                
                EPOCH_DISPLAY = int( int(results_val['global_step']) / (TRAIN_SIZE/BATCH_SIZE))
                print('Epoch = {}; Step = {} / ValLoss = {:.5f};'.format(
                     EPOCH_DISPLAY, 
                     results_val['global_step'], 
                     results_val['loss']))
                
                dim                        = args.model_path + 'Step{}ValLoss{:.5f}'.format(results_val['global_step'], results_val['loss'])
                export_dir                 = nn.export_savedmodel(
                export_dir_base            = dim,
                serving_input_receiver_fn  = reader.serving_input_receiver_fn(reader_patch_shapes))
                print('Model saved to {}.'.format(export_dir))
                count_steps.append(results_val['global_step'])
                count_loss.append(results_val['loss'])

    except KeyboardInterrupt:
        pass

    # Arbitrary Input Shape during Export
    export_dir = nn.export_savedmodel(
        export_dir_base           = args.model_path,
        serving_input_receiver_fn = reader.serving_input_receiver_fn(reader_patch_shapes))
    print('Model saved to {}.'.format(export_dir))

    step_Loss      = pd.DataFrame(list(zip(count_steps,count_loss)),
    columns        = ['steps','val_loss'])
    step_Loss.to_csv("Validation_Loss.csv", encoding='utf-8', index=False)



if __name__ == '__main__':

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Argument Parser Setup
    parser = argparse.ArgumentParser(description='Multi-Class Brain Segmentation')
    parser.add_argument('--run_validation',     default=True)
    parser.add_argument('--restart',            default=False, action='store_true')
    parser.add_argument('--verbose',            default=False, action='store_true')
   
    parser.add_argument('--cuda_devices', '-c', default='0')
    parser.add_argument('--model_path',   '-p', default='/home/cad1/anindo/neuro/model_weights/deepmedic01/')
    parser.add_argument('--train_csv',    '-t', default='/home/cad1/anindo/neuro/feed/Training-Fold.csv')
    parser.add_argument('--val_csv',      '-v', default='/home/cad1/anindo/neuro/feed/Validation-Fold.csv')

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

    # Handle Restarting/Resuming Training
    if args.restart:
        print('Restarting training from scratch.')
        os.system('rm -rf {}'.format(args.model_path))

    if not os.path.isdir(args.model_path):
        os.system('mkdir -p {}'.format(args.model_path))
    else:
        print('Resuming training on model_path {}'.format(args.model_path))

    # Train
    train(args)

    session.close()

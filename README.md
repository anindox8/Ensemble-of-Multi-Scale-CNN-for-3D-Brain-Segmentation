# Ensemble of Convolutional Neural Networks for 3D Multi-Class Brain Segmentation in T1 MRI

**Problem Statement**: Fully supervised, multi-class 3D brain segmentation in T1 MRI. 

**Note**: The following approach won __ place in the **2019 Medical Image Segmentation and Applications: Brain Tissue Segmentation Challenge** at [Universitat de Girona](https://www.udg.edu) scoring **______ at test-time**, during the 2018-20 Joint Master of Science in [Medical Imaging and Applications (MaIA)](https://maiamaster.udg.edu) program.  

**Acknowledgments**: DLTK for the TensorFlow.Estimator implementation of [3D U-Net, 3D FCN and DeepMedic](https://github.com/DLTK/models) model architectures and NiftyNet for the  TensorFlow implementation of [Cross-Entropy and Dice Loss](https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py).
 
**Data**: *Label 0*: Background; *Label 1*: Cerebrospinal Fluid (CSF); *Label 2:* Gray Matter (GM); *Label 3:* White Matter (GM) [10/5/3 : Train/Val/Test Ratio]
 
 
**Directories**  
  ● Resample NIfTI Volume Resolutions: `scripts/preprocess.py`  
  ● Generate Data-Directory Feeder List: `scripts/feed_io.py`  
  ● Train Residual 3D FCN: `scripts/resfcn_train-val.py`  
  ● Train Residual 3D U-Net: `scripts/resunet_train-val.py`  
  ● Train DeepMedic: `scripts/deepmedic_train-val.py` *(Discontinued)*  
  ● Deploy Ensemble Model (Validation/Testing): `scripts/deploy_e.py` 

  
## Dataset  

*Table 1.  *
![Data Augmentation](reports/images/data_augmentation.png)
   
     
## Patch Extraction/Multi-Scale Input  

![Multi-Scale Input](reports/images/multi-scale_io.png)*Figure 1.  * 

    
## Loss Function 

![Feature Maps](reports/images/imgnet_efn.png)*Figure 2.  *  
 

## Effect of Preprocessing 

*Table 2.  *
![Results](reports/images/results.png) 


## Model Performance  

*Table 3.  *
![GradCAM](reports/images/gradcam.png)



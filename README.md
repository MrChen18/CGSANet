# [CGSANet: A contour-guided and local structure-aware encoder-decoder network for accurate building extraction from very high-resolution remote sensing imagery](https://ieeexplore.ieee.org/document/9664368)

January 4th, 2022

## Introduction

Extracting buildings accurately from very high-resolution (VHR) remote sensing imagery is challenging due to diverse building appearances, spectral variability, and complex background in VHR remote sensing images. Recent studies mainly adopt a variant of the Fully Convolutional Network (FCN) with an encoder-decoder architecture to extract buildings, which has shown promising improvement over conventional methods. However, FCN-based encoder-decoder models still fail to fully utilize the implicit characteristics of building shapes. This adversely affects the accurate localization of building boundaries, which is particularly relevant in building mapping. A contour-guided and local structure-aware encoder-decoder network (CGSANet) is proposed to extract buildings with more accurate boundaries. CGSANet is a multi-task network composed of a contour-guided (CG) and a multi-region guided (MRG) module. The CG module is supervised by a building contour that effectively learns building contour-related spatial features to retain the shape pattern of buildings. The MRG module is deeply supervised by four building regions that further capture multi-scale and contextual features of buildings. In addition, a hybrid loss function was designed to improve the structure learning ability of CGSANet. These three improvements benefit each other synergistically to produce high-quality building extraction results. Experiment results on the WHU and NZ32km2 building datasets demonstrate that compared with the tested algorithms, CGSANet can produce more accurate building extraction results and achieve the best intersection over union (IoU) value 91.55% and 90.02%, respectively. Experiments on the INRIA building dataset further demonstrate the ability for generalization of the proposed framework, indicating great practical potential.

## Prerequisites

scikit-image  0.17.2  
numpy  1.19.3  
Python3 Python 3.6.2  
PyTorch  torch 1.7.0+cu110  

## Architecture

![CGSANet ](https://github.com/MrChen18/CGSANet/blob/main/figures/Architecture_CGSANet.jpg "CGSANet")

## Boundary accuracy evaluation on WHU dataset

![Boundary evaluation ](https://github.com/MrChen18/CGSANet/blob/main/figures/Boundary%20accuracy%20evaluation%20on%20WHU%20dataset.jpg "Boundary evaluation")

## Pretrained Model

The pretrained models for WHU aerial building dataset and INRIA dataset can be download from [Google Drive](https://drive.google.com/drive/folders/1LD49DUJ9cw9DX7ssow2CQGZLpktvg8tZ?usp=sharing), [Baidu Wangpan](https://pan.baidu.com/s/1WWtrBzGmbM2eoO8NV1_ybg?pwd=nfyx )提取码: nfyx. 

## Datasets

[WHU aerial dataset](http://gpcv.whu.edu.cn/data/building_dataset.html)  
[Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)  
[NZ32km2 dataset](https://drive.google.com/file/d/1PNkGLRT8J9h4Cx9iyS0Bh9vamQS_KOTz/view)

## Acknowledgement

We appreciate the work from the following repositories: 
[BASNet](https://github.com/xuebinqin/BASNet) and [F3Net](https://github.com/weijun88/F3Net).

## License

This code is available for non-commercial scientific research purposes under GNU General Public License v3.0. By downloading and using this code you agree to the terms in the LICENSE. Third-party datasets and software are subject to their respective licenses.

## Citation

If you find this work is helpful, please cite our paper

```
@ARTICLE{9664368,
  author={Chen, Shanxiong and Shi, Wenzhong and Zhou, Mingting and Zhang, Min and Xuan, Zhaoxin},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={CGSANet: A Contour-Guided and Local Structure-Aware Encoder–Decoder Network for Accurate Building Extraction From Very High-Resolution Remote Sensing Imagery}, 
  year={2022},
  volume={15},
  number={},
  pages={1526-1542},
  doi={10.1109/JSTARS.2021.3139017}}
```

# DecoupleSegNets
This repo contains the full implementation of Our ECCV-2020 work: Improving Semantic Segmentation viaDecoupled Body and Edge Supervision.

This is the join work of Peking University, University of Oxford and Sensetime Research. (Much thanks for Sensetimes' GPU server)


Any Suggestions and Questions are welcome. I will reply it as soon as possible.
It also contains reimplementation of our previous AAAI-2020 work(oral)  
GFFNet:Gated Fully Fusion for semantic segmentation which also achieves the state-of-the-art results on CityScapes:
  

![avatar](./fig/teaser.png)


# DataSet preparation
Dataloaders for Cityscapes, Mapillary, Camvid ,BDD and Kitti are available in [datasets](./datasets). 
Details of preparing each dataset can be found at [PREPARE_DATASETS.md](https://github.com/lxtGH/DecoupleSegNets/blob/master/DATASETs.md)


# Model Checkpoint


## Pre-trained Models


## Trained Models




# Training

To be note that, Our best models(Wider-ResNet-38) are trained on 8 V-100 GPUs with 32GB memory.
        It is hard to reproduce such best results if you do not have such resources.
However, our resnet-based methods including fcn, deeplabv3+, pspnet can be trained by 8-1080-TI gpus witg batchsize 16.
Our training contains two steps(Here I give the ):


## 1, Train the base model.

You can find our pretrained model in the following links.

## 2, Re-Train with our module with lower LR using pretrained models.


### For DecoupleSegNets:


### For GFFNet:


# Evaluation

## 1, Single-Scale Evaluation
```bash

```

## 2, Multi-Scale Evaluation
```bash

```

## 3, Evaluate F-score on Segmentation Boundary.(change the path of snapshot)
```bash
sh ./scripts/evaluate_boundary_fscore/evaluate_cityscapes_deeplabv3_r101_decouple
```

# Submission on Cityscapes



# Demo 
Here we give some demo scripts for using our checkpoints.
You can change the scripts according to your needs.

```bash

```

# Citation
If you find this repo is helpful to your research. Please consider cite our work.



# Acknowledgement
This repo is based on NVIDIA segmentation repo. We fully thank their open-sourced code.


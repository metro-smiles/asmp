# Overview
Code repository for ["Learning Audio-Visual Dynamics Using Scene Graphs for Audio Source Separation"](https://arxiv.org/pdf/2210.16472.pdf), NeurIPS 2022. 

[[Project Page]](https://sites.google.com/site/metrosmiles/research/research-projects/asmp)

<br/>

<img src='Arch_Fig.png' align="left" >

<br/>

Intelligent systems need to draw meaningful deductions about objects in a scene by associating their visual appearance and motion with their audio signatures.

In this work, we consider the task of visually guided audio source separation and use this separated audio (derived from the visual conditioning information) to coarsely estimate the direction of motion of the sound source. Towards this end, we propose a deep neural network, Audio Separator and Motion Predictor (ASMP), with the following three architectural components:

 * Visual Conditioining Module
 * Audio-Separator Network
 * Direction Prediction Network

As illustrated in the figure above, ASMP begins by leveraging its Visual Conditioning module to create graph embeddings of potential auditory sources and their context nodes. Towards this end, this module employs Graph Ateention Networks and Edge Convolution. Importantly the graph construction encodes scene geometry information. Next, the graph embeddings are used to condition a U-Net style network responsible for undertaking the audio source separation, called the Audio Separator Network. Finally this conditionally separated output is passed through a direction prediction network, to estimate the direction of motion (one of 28 classes).

# Key Libraries
Please refer to requirements.txt file.

# Datasets
In the paper, we report experimental results on the [ASIW](https://sites.google.com/site/metrosmiles/research/research-projects/avsgs) dataset among others.

Please download the pre-processed Audio Separation in the Wild (ASIW) dataset from this [link](https://data.aifarms.org/files/links/6356f2a5e4b04f23bdc31bd5). Then, unzip it into the root directory where this repository is cloned. 

Then download the pre-computed pairwise Chamfer Distances and the object displacement vectors from this [link](). Make sure to put the Rectified_Frames folder under *<path to>/Audiocaps_Dataset/*

# Pre-trained Models
The pre-trained models for this task may be downloaded from here: [link]()


# Evaluation
The evaluation for the audio source separation may be performed by running the following commands:

```
$ export MODEL_PATH="<path to root>/checkpoints/pretrained_models/"
$ python -W ignore test_batch_noaug_varcontxt_full300_onlysep.py --visual_pool conv1x1 --unet_num_layers 7 --data_path <path to>/Audiocaps_Dataset/ --hdf5_path <path to>/audiocaps/dataset/ --gpu_ids 0 --weights_unet $MODEL_PATH/unet_final.pth  --weights_classifier $MODEL_PATH/classifier_final.pth --weights_graph $MODEL_PATH/graph_net_final.pth --weights_map_net $MODEL_PATH/map_net_final.pth --weights_rnn $MODEL_PATH/rnn_final.pth --num_of_object_detections_to_use 1 --with_additional_scene_image --sigma 25 --scene_path ./pre_trained/ADE.h5 --output_dir_root ./test_results/
```

The evaluation for the direction prediction task may be performed by running the following command:

```
$ export MODEL_PATH="<path to root>/checkpoints/pretrained_models/"
$ python -W ignore test_batch_noaug_varcontxt_dirpred_fin28class.py --visual_pool conv1x1 --unet_num_layers 7 --data_path <path to>/Audiocaps_Dataset/ --hdf5_path <path to>/audiocaps/dataset/ --gpu_ids 0 --weights_unet $MODEL_PATH/unet_final.pth  --weights_classifier $MODEL_PATH/classifier_final.pth --weights_graph $MODEL_PATH/graph_net_final.pth --weights_map_net $MODEL_PATH/map_net_final.pth --weights_rnn $MODEL_PATH/rnn_final.pth --weights_fuse_enc $MODEL_PATH/fusion_enc_final.pth --weights_net_fuse $MODEL_PATH/dir_classifier_final.pth --num_of_object_detections_to_use 1 --with_additional_scene_image --number_of_dir_classes 28 --sigma 25 --scene_path ./pre_trained/ADE.h5 --output_dir_root ./test_results/
```

# Citation
If you find our code or data useful in your research, please cite either this work or our previous work or both:

     @inproceedings{chatterjee2021visual,
      title={Learning Audio-Visual Dynamics Using Scene Graphs for Audio Source Separation},
      author={Chatterjee, Moitreya and Ahuja, Narendra and Cherian, Anoop},
      booktitle={Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)},
      year={2022}
    }
    
     @inproceedings{chatterjee2021visual,
      title={Visual Scene Graphs for Audio Source Separation},
      author={Chatterjee, Moitreya and Le Roux, Jonathan and Ahuja, Narendra and Cherian, Anoop},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      pages={1204--1213},
      year={2021}
    }

# Acknowledgement
Thanks to [Ruohan Gao](https://ai.stanford.edu/~rhgao/), for inspiring this work.

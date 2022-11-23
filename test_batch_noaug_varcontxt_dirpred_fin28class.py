#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import h5py
import os
import argparse
import librosa
import pickle
import numpy as np
from PIL import Image
import csv
import random

import torchvision.transforms as transforms
import torch

from random import randrange
import torch.nn.functional as F
import torch.nn as nn
from options.test_options import TestOptions
from models.models import ModelBuilder
from models.audioVisual_model_noaug_noclassif import AudioVisualModel
from data.audioVisual_dataset import generate_spectrogram_magphase, compute_overlap
from utils import utils
from mir_eval.separation import bss_eval_sources
import ast

# Use 28 classes
disp_cls = {str([0,0,0]): 0, str([0,0,1]): 1, str([0,1,0]): 2, str([1,0,0]): 3, str([0,0,-1]): 4, str([0,-1,0]): 5, str([-1,0,0]): 6, \
    str([0,1,1]): 7, str([1,1,0]): 8, str([1,0,1]): 9, str([0,1,-1]): 10, str([1,-1,0]): 11, str([1,0,-1]): 12, str([-1,1,0]): 13, str([-1,0,1]): 14, \
    str([0,-1,1]): 15, str([0,-1,-1]): 16, str([-1,-1,0]): 17, str([-1,0,-1]): 18, str([1,1,1]): 19, str([-1,1,1]): 20, str([-1,-1,1]): 21, \
    str([1,-1,1]): 22, str([1,1,-1]): 23, str([-1,1,-1]): 24, str([-1,-1,-1]): 25, str([1,-1,-1]): 26}
# Define the thresholds for considering no displacement
u_thresh, v_thresh, d_thresh = 0.60, 0.60, 0.30
# Set the first frame of every window
img_obj_wind = {0: "0001.png", 1: "0009.png", 2: "0017.png", 3: "0025.png", 4: "0033.png", 5: "0041.png", 6: "0049.png", 7: "0057.png", 8: "0065.png", 9: "0073.png"}
# Define the filename containing the pairs to be ealuated on
pair_fl = 'DirPred_PairList.pickle' 

def sample_audio(audio, window, sr, obj_win):
    # repeat if audio is too short
    if audio.shape[0] < window:
        n = int(window / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    max_strt = np.int_((audio.shape[0] - window) // sr) + 1
    audio_start = 0 #*random.choice(np.arange(max_strt)) * sr #* randrange(0, audio.shape[0] - window + 1)
    audio_end = audio_start+window

    # Repeat until the start doesn't include the beginning of the sub_window
    while (audio_start > (obj_win * sr)) or (audio_end < (obj_win * sr)): #* or (audio_end < sub_window[1]):
        audio_start = audio_start + sr #* random.choice(np.arange(max_strt)) * sr #* randrange(0, audio.shape[0] - window + 1)
        audio_end = audio_start+window

    audio_sample = audio[audio_start:(audio_start+window)]

    return audio_sample, obj_win - (audio_start // sr) # Return the sampled audio and the new index of the window

def clip_audio(audio):
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def get_separated_audio(outputs, batch_data, opt):
    # fetch data and predictions
    mag_mix = batch_data['audio_mix_mags']
    phase_mix = batch_data['audio_mix_phases']
    pred_masks_ = outputs['pred_mask']
    mag_mix_ = outputs['audio_mix_mags']
    # unwarp log scale
    B = mag_mix.size(0)
    if opt.log_freq:
        grid_unwarp = torch.from_numpy(utils.warpgrid(B, opt.stft_frame//2+1, pred_masks_.size(3), warp=False)).to(opt.device) #* 
        pred_masks_linear = F.grid_sample(pred_masks_, grid_unwarp)
    else:
        pred_masks_linear = pred_masks_
    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    pred_masks_linear = pred_masks_linear.detach().cpu().numpy()
    pred_mag = mag_mix[0, 0] * pred_masks_linear[0, 0]
    preds_wav = utils.istft_reconstruction(pred_mag, phase_mix[0, 0], hop_length=opt.stft_hop, length=opt.audio_window)
    return preds_wav

def getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt):
        reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0)), axis=0)
        #print reference_sources.shape
        estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
        #print estimated_sources.shape
        (sdr, sir, sar, perm) = bss_eval_sources(np.asarray(reference_sources), np.asarray(estimated_sources), False)
        #print sdr, sir, sar, perm
        return np.mean(sdr), np.mean(sir), np.mean(sar)

def graph_declare(builder, opt, model_nm):
    return builder.build_graph_encoder( 
    feat_dim=512, heads=4, hidden_act=opt.hidden_act, fin_graph_rep=opt.graph_enc_dim, 
    pooling_ratio=opt.pooling_ratio, nos_classes=opt.number_of_classes, gnet_type=opt.gnet_type, weights=model_nm) 

def get_chamfer_info(vid_info, opt):
    vid_princip = {} # Store principal object and context node information
    # Define the list of frame and bounding box information
    obj_con, p_i = {}, 0
    for o_c in vid_info['object_context'][:-1]:
        p_o_ind = o_c[2]
        if p_o_ind == 1: # If it is a principal object
            # Check if its window is -1
            vis_obj, obj_wind = vid_info['object_wind'][p_i]
            #*if obj_wind == -1:
            vid_princip[vid_info['object_wind'][p_i]] = (o_c[0], o_c[1]) # Store the im_id, and box index in frame
            obj_con[vid_info['object_wind'][p_i]] = [] # Initialize the list
            p_i += 1

    p_i = 0 # Reinitialize the principal object counter
    for o_c in vid_info['object_context'][:-1]:
        p_o_ind = o_c[2]
        if p_o_ind == 1: # If it is a principal object
            p_i += 1            
        else: # If it is a context object
            obj_con[vid_info['object_wind'][p_i]].append(o_c)
    
    # Load the Chamfer distance matrix
    chamf_dist = vid_info['dist_mat']
    # Design appropriate edge matrix
    edge_wt = chamf_dist.reshape(-1)
    max_d = edge_wt[~np.isnan(edge_wt)].max()
    # Filter out the non-zero elements from the edge-weight matrix
    nz_edge_wt = edge_wt[np.nonzero(edge_wt)].reshape(-1)
    # Check if sufficient number of non-zero elements exist
    if nz_edge_wt.shape[0] > 0:
        med_wt = np.percentile(nz_edge_wt, opt.sigma) #* np.median(e_w[np.nonzero(e_w)]) # This returns the median value of the non-zero elements of the edge weights
        # Define the RBF Kernel
        chamf_dist = np.exp((vid_info['dist_mat']/max_d) * -1./ med_wt)
    else:
        med_wt = 0
        chamf_dist = np.ones_like(vid_info['dist_mat'])

    return vid_princip, obj_con, chamf_dist

def build_feature_tensor(vid_info, vid_princip, vid_meta, obj_det, obj_con, disp, disp_wind, opt):
    person_visuals = []
    # Only keep num_object_per_video sound sources
    for i in range(min(len(vid_info['object_wind']), opt.num_object_per_video)): # Iterate over each source detected
        # Choose a window for the current object - randomly
        vis_obj, obj_wind = vid_info['object_wind'][i] 

        if obj_wind == -1: # No displacement
            im_id, obj_frm_idx = vid_princip[(vis_obj, obj_wind)]
        else:
            img_name = str(img_obj_wind[obj_wind]) # Extract visual info from first frame of selected window 
            im_id = [i_ for i_ in range(len(vid_meta)) if vid_meta[i_]['image'] == img_name][0]  # Get the im_id for the matched frame
            # Since displacement pickle contains current window, so object must be present in first frame
            obj_frm_idx = vid_meta[im_id]['objects'].index(int(vis_obj)) # Find the index of the sounding object from the list of objects in the first frame of the window

        # Get the list of context nodes
        con_lst = obj_con[(vis_obj, obj_wind)]
        
        # Extract the context information for the current 
        if len(con_lst) > 0:
            # If atleast one context node is available, then include all valid context nodes
            for o_c in con_lst:
                c_im_id, p, _ = o_c
                person_visuals.append(torch.from_numpy(obj_det['features'][c_im_id, p, :]).view(1, -1)) # Obtain the context node features

        # Incorporate the principal object information
        person_visuals.append(torch.from_numpy(obj_det['features'][im_id, obj_frm_idx, :]).view(1, -1)) # Obtain the sound node features

        # Compute displacement vectors
        disp_vec = disp[int(vis_obj)][disp_wind]
        #*print('Displacement vector: ' + str(disp_vec))
        # Check if the vector should go to class 0, if not then normalize
        if (np.abs(disp_vec[0]) >= u_thresh) or (np.abs(disp_vec[1]) >= v_thresh) or (np.abs(disp_vec[2]) >= d_thresh): #* Old check: np.sum(disp_vec) != 0:
            basis_vecs_str, basis_vecs = list(disp_cls.keys()), []
            for _ in basis_vecs_str:
                tmp = np.array(ast.literal_eval(_)) #* exec('tmp = np.array({})'.format(_))
                basis_vecs.append(tmp)

            basis_vecs = np.array(basis_vecs)
            # Compute the norm of the basis vectors
            norm_mat =  np.linalg.norm(basis_vecs, axis=-1)
            # Convert the basis vectors to unit norm, skip (0,0,0) class
            basis_vecs =  basis_vecs[1:, :] / np.tile(norm_mat[1:].reshape(-1, 1), (1, 3))
            # Convert the displacement vector to unit norm
            disp_vec = disp_vec / np.linalg.norm(disp_vec)
            # Compute cosine similarity
            sim = np.dot(basis_vecs, disp_vec)
            # Add 1 to account for the previous discarding of (0, 0, 0)
            max_idx = np.argmax(sim) + 1 
            # Final alignment
            u_v = basis_vecs_str[max_idx]
            # Introduce the label
            dir_v = disp_cls[u_v]
        else:
            # Implies there is no motion
            dir_v = disp_cls[str([0,0,0])]

    # Augment the background features when applicable
    if opt.with_additional_scene_image:  
        person_visuals.append(torch.from_numpy(obj_det['features'][0, 0, :]).view(1, -1)) # Choose the first box of the first im_id as the random object

    return person_visuals, dir_v

def build_edge_tensor(person_visuals, chamf_dist):
    # Check if person_visuals is not only object, then construct edge tensor
    if person_visuals.size(0) == 1:
        edges = torch.Tensor([[0], [0]]).long() 
        edges_attr = torch.Tensor([1])
        batch_vec = torch.Tensor([0]).long()
    else:
        sub, obj, sub_f, obj_f, edges_attr, bv = [], [], [], [], [], []
        for nd1 in range(person_visuals.size(0)):
            for nd2 in range(person_visuals.size(0)):
                # Check if Chamfer distance matrix is the same shape as current node
                if (chamf_dist.shape[0] == person_visuals.size(0)): #* opt.median_weights and 
                    if (nd1 < nd2): # Upper traiangular matrix part
                        edges_attr.append(chamf_dist[nd1, nd2])
                    elif (nd2 < nd1): # Lower traiangular matrix part
                        edges_attr.append(chamf_dist[nd2, nd1])
                    else:
                        edges_attr.append(1)  
                else:
                    edges_attr.append(1)  

                # Construct edges for the full graph
                sub.append(nd1)
                obj.append(nd2)

            # Incorporate batch_vec info
            bv.append(0)
        # Convert the lists to tensors
        edges = torch.Tensor([sub, obj]).long()
        edges_attr = torch.Tensor(edges_attr)
        batch_vec = torch.Tensor(bv).long()

    return edges, edges_attr, batch_vec

def main():
    #load test arguments
    opt = TestOptions().parse()
    opt.device = torch.device("cuda")
    print(opt.output_dir_root)
    opt.output_dir_root = opt.output_dir_root.rstrip()
    # Network Builders
    builder = ModelBuilder()
    net_visual = None 

    net_unet = builder.build_unet(
            unet_num_layers = opt.unet_num_layers,
            ngf=opt.unet_ngf,
            input_nc=opt.unet_input_nc,
            output_nc=opt.unet_output_nc,
            weights=opt.weights_unet)
    
    if opt.with_additional_scene_image:
        opt.number_of_classes = opt.number_of_classes + 1

    graph_nets = graph_declare(builder, opt, opt.weights_graph) # Define the graph encoder

 
    map_net = nn.Sequential(nn.Linear(opt.feat_dim, opt.feat_dim // 2), nn.LeakyReLU(negative_slope=0.2), \
    nn.Linear(opt.feat_dim // 2, 512)) # Define the mapping network for person nodes

    map_net.load_state_dict(torch.load(opt.weights_map_net)) # Load the weights from the saved model

    # Define the recurrent layer
    rnn = nn.GRU(input_size=512, hidden_size=512, num_layers=1)
    rnn.load_state_dict(torch.load(opt.weights_rnn)) # Load the weights from the saved model

    # Define the RNN classifier
    rnn_classif = None 

    # Define the network for the spectrogram encoding for fusion
    net_fusion_enc = builder.build_classifier(pool_type=opt.classifier_pool, num_of_classes=rnn.hidden_size,
       input_channel=opt.unet_output_nc, weights=opt.weights_fuse_enc)

    # Check if the graph encoder is to be used
    if opt.use_graph_enc:
        
        # Define the network for fusing the spectrogram and the visual cues
        net_fuse = nn.Sequential(nn.Linear(2 * rnn.hidden_size, rnn.hidden_size), nn.BatchNorm1d(rnn.hidden_size), nn.LeakyReLU(negative_slope=0.2), \
            nn.Linear(rnn.hidden_size, rnn.hidden_size // 2), nn.BatchNorm1d(rnn.hidden_size // 2), nn.LeakyReLU(negative_slope=0.2), \
            nn.Linear(rnn.hidden_size // 2, opt.number_of_dir_classes), nn.BatchNorm1d(opt.number_of_dir_classes))

    else:
        # Define the network for fusing the spectrogram and the visual cues
        net_fuse = nn.Sequential(nn.Linear(rnn.hidden_size, rnn.hidden_size // 2), nn.BatchNorm1d(rnn.hidden_size // 2), nn.LeakyReLU(negative_slope=0.2), \
            nn.Linear(rnn.hidden_size // 2, opt.number_of_dir_classes), nn.BatchNorm1d(opt.number_of_dir_classes))

    net_fuse.load_state_dict(torch.load(opt.weights_net_fuse)) # Load the weights from the saved model
    
    nets = (None, net_unet, None, graph_nets, map_net, rnn, rnn_classif, net_fusion_enc, net_fuse) #* 

    # construct our audio-visual model
    model = AudioVisualModel(nets, opt)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    model.to(opt.device)
    model.eval()

    rnn_init = torch.zeros(rnn.num_layers, 1, rnn.hidden_size).cuda(opt.gpu_ids[0]) # Assume batchSize of 1 here
    print('All models loaded!')

    # Load the pairs on which to evaluate
    with open(os.path.join(opt.output_dir_root, pair_fl), 'rb') as f: 
        pairs_lst_full, _, _, _ = pickle.load(f)

    pair_lst = list(pairs_lst_full.keys())
    
    print('Number of pairs: ' + str(len(pair_lst)))
    # Stores the results
    perf_dict, res_dict, missed_seq, sparsity = {}, {}, [], []
    # Iterate over each pair of videos
    for pr in pair_lst: # Iterate over the list of source videos
            v1, c1, obj_wind_1, v2, c2, obj_wind_2 = pr # Obtain the video names from the pair 
            
            #load the two audios
            audio1_path = os.path.join(opt.data_path, 'Extracted_Audio/test/', v1 + '.aac') 
            audio1_f, _ = librosa.load(audio1_path, sr=opt.audio_sampling_rate)
            audio2_path = os.path.join(opt.data_path, 'Extracted_Audio/test/', v2 + '.aac') 
            audio2_f, _ = librosa.load(audio2_path, sr=opt.audio_sampling_rate)

            # Load the displacements
            with open(os.path.join(opt.data_path, 'Rectified_Frames/test/', v1, 'Comb_RAFTFlow_Displacement_Vectors_New.pickle'), 'rb') as f:
                disp_1 = pickle.load(f)

            with open(os.path.join(opt.data_path, 'Rectified_Frames/test/', v2, 'Comb_RAFTFlow_Displacement_Vectors_New.pickle'), 'rb') as f:
                disp_2 = pickle.load(f)

            # Load the json file for the first video
            with open(os.path.join(opt.data_path, 'Extracted_Frames/test/', v1, 'Image_info_20obj.json'), 'r') as f:
                vid_meta_1 = json.load(f)

            # Load the json file for the second video
            with open(os.path.join(opt.data_path, 'Extracted_Frames/test/', v2, 'Image_info_20obj.json'), 'r') as f:
                vid_meta_2 = json.load(f)

            # Determine the list of objects for the two videos
            obj_lst_1 = list(disp_1.keys())
            obj_lst_2 = list(disp_2.keys())

            print('Currently processing pairs: ' + str(v1) + ', ' + str(c1) + ', ' + str(v2) + ', ' + str(c2))
            
            # Open the h5py file for man/woman node for the first video
            obj_det_1 = h5py.File(os.path.join(opt.data_path, 'Extracted_Frames/test/', v1, 'Image_feature_20obj.h5'), 'r')

            # Open the h5py file for man/woman node for the second video
            obj_det_2 = h5py.File(os.path.join(opt.data_path, 'Extracted_Frames/test/', v2, 'Image_feature_20obj.h5'), 'r')

            # Adjust the lengths of the two audio signals to make them full length
            audio1, rel_obj_wind_1 = sample_audio(audio1_f, opt.audio_window, opt.audio_sampling_rate, obj_wind_1)
            audio2, rel_obj_wind_2 = sample_audio(audio2_f, opt.audio_window, opt.audio_sampling_rate, obj_wind_2)

            # Load the pre-computed Chamfer distance for the first video
            if os.path.isfile(os.path.join(opt.data_path, 'Rectified_Frames/test/', v1, 'Chamfer_Info.pickle')):
                with open(os.path.join(opt.data_path, 'Rectified_Frames/test/', v1, 'Chamfer_Info.pickle'), 'rb') as f:
                    vid_info_1 = pickle.load(f)
            else:
                with open(os.path.join(opt.data_path, 'Rectified_Frames/test/', v1, 'Chamfer_Info_Prev.pickle'), 'rb') as f:
                    vid_info_1 = pickle.load(f)

            # Load the pre-computed Chamfer distance for the second video
            if os.path.isfile(os.path.join(opt.data_path, 'Rectified_Frames/test/', v2, 'Chamfer_Info.pickle')):
                with open(os.path.join(opt.data_path, 'Rectified_Frames/test/', v2, 'Chamfer_Info.pickle'), 'rb') as f:
                    vid_info_2 = pickle.load(f)
            else:
                with open(os.path.join(opt.data_path, 'Rectified_Frames/test/', v2, 'Chamfer_Info_Prev.pickle'), 'rb') as f:
                    vid_info_2 = pickle.load(f)

            # Store the essential visual information for the first video
            vid_princip_1, obj_con_1, chamf_dist_1 = get_chamfer_info(vid_info_1, opt)
            #*print('get_chamfer_info(): ' + str(vid_princip_1) + ' ' + str(obj_con_1) + ' ' + str(chamf_dist_1))

            # Store the essential visual information for the second video
            vid_princip_2, obj_con_2, chamf_dist_2 = get_chamfer_info(vid_info_2, opt)
            #*print('get_chamfer_info(): ' + str(vid_princip_2) + ' ' + str(obj_con_2) + ' ' + str(chamf_dist_2))

            #make sure the two audios are of the same length and then mix them
            audio_length = min(len(audio1), len(audio2))
            audio1 = clip_audio(audio1[:audio_length])
            audio2 = clip_audio(audio2[:audio_length])
            audio_mix = (audio1 + audio2) / 2.0
            #*print('Length of mixed audio: ' + str(audio_length))

            # Initialize the separated audios
            avged_sep_audio1 = np.zeros((audio_length)) #* np.zeros((audio_win_len)) #*
            avged_sep_audio2 = np.zeros((audio_length)) #* np.zeros((audio_win_len)) #*

            for i in range(opt.num_of_object_detections_to_use):
                # Get the visual features for the first video
                person_visuals_1, dir_1 = build_feature_tensor(vid_info_1, vid_princip_1, vid_meta_1, obj_det_1, obj_con_1, disp_1, obj_wind_1, opt)
                
                # Get the visual features for the second video
                person_visuals_2, dir_2 = build_feature_tensor(vid_info_2, vid_princip_2, vid_meta_2, obj_det_2, obj_con_2, disp_2, obj_wind_2, opt)
                #*print('build_feature_tensor(): ' + str(person_visuals_2) + ' ' + str(dir_2))
                
                # Convert the feature lists into tensors
                person_visuals_1 = torch.cat(person_visuals_1, dim=0)
                person_visuals_2 = torch.cat(person_visuals_2, dim=0)
                #*print('Shape of feature tensor: ' + str(person_visuals_1.shape) + ', ' + str(person_visuals_2.shape))
                
                # Get the edge information for the first graph
                edges_1, edges_attr_1, batch_vec_1 = build_edge_tensor(person_visuals_1, chamf_dist_1)
                #*print('build_edge_tensor(): ' + str(edges_1) + ' ' + str(edges_attr_1) + ' ' + str(batch_vec_1))

                # Get the edge information for the second graph
                edges_2, edges_attr_2, batch_vec_2 = build_edge_tensor(person_visuals_2, chamf_dist_2)
                #*print('build_edge_tensor(): ' + str(edges_2) + ' ' + str(edges_attr_2) + ' ' + str(batch_vec_2))

                #perform separation over the selected audio window length using a sliding window approach
                overlap_count = np.zeros((audio_length)) #* np.zeros((audio_win_len)) #*
                sep_audio1 = np.zeros((audio_length)) #* np.zeros((audio_win_len)) #*
                sep_audio2 = np.zeros((audio_length)) #* np.zeros((audio_win_len)) #*
                sliding_window_start = 0
                corr_flg1, corr_flg2 = 0, 0 # The flags indicating correct octant or not
                wind_size, overlap_len = 46, 4 
                shift_wind, data = wind_size - overlap_len, {}
                samples_per_window = opt.audio_window #* audio_win_len #* 
                while sliding_window_start + samples_per_window < audio_length: 
                    sliding_window_end = sliding_window_start + samples_per_window
                    audio_segment = audio_mix[sliding_window_start:sliding_window_end]
                    audio_mix_mags, audio_mix_phases = generate_spectrogram_magphase(audio_segment, opt.stft_frame, opt.stft_hop) 
                    data['audio_mix_mags'] = torch.FloatTensor(audio_mix_mags).unsqueeze(0)
                    data['audio_mix_phases'] = torch.FloatTensor(audio_mix_phases).unsqueeze(0)
                    data['real_audio_mags'] = data['audio_mix_mags'] #dont' care for testing
                    data['audio_mags'] = data['audio_mix_mags'] #dont' care for testing
                    #separate for video 1
                    data['per_visuals'] = person_visuals_1
                    data['edges'] = edges_1
                    data['edge_wt'] = edges_attr_1
                    data['batch_vec'] = batch_vec_1
                    data['labels'] = torch.FloatTensor(np.ones((1,1)))#don't care for testing torch.Tensor([det_box1[1]]).long() #* 
                    data['vids'] = torch.FloatTensor(np.ones((1,1)))#don't care for testing
                    data['dir_labels'] = torch.FloatTensor(dir_1) # Direction Label
                    outputs = model.forward(data, rnn_init=rnn_init, mode=opt.mode, step='reconstruct')
                    data['visual_feature'] = outputs['visual_feature']
                    
                    # Create the inputs for the direction prediction module
                    strt_wind_1, end_wind_1 = rel_obj_wind_1 * shift_wind, (rel_obj_wind_1 * shift_wind) + wind_size
                    data['real_audio_mags'] = outputs['pred_spectrogram'][:, :, :, strt_wind_1 : end_wind_1] # Separated spectrogram
                    data['audio_mags'] = outputs['pred_spectrogram'][:, :, :, strt_wind_1 : end_wind_1] # Separated spectrogram
                    # Forward pass through the direction prediction module
                    outputs = model.forward(data, rnn_init=rnn_init, mode=opt.mode, step='dir_pred')
                    # Obtain the direction label
                    pred_label_1 = F.softmax(outputs['pred_dir'], dim=-1).squeeze().argmax().item()
                    if pred_label_1 == dir_1: # If labels match, then update flag
                        corr_flg1 = 1
                    
                    #separate for video 2
                    data['per_visuals'] = person_visuals_2
                    data['edges'] = edges_2
                    data['edge_wt'] = edges_attr_2
                    data['batch_vec'] = batch_vec_2
                    data['labels'] = torch.FloatTensor(np.ones((1,1)))#don't care for testing torch.Tensor([det_box2[1]]).long() #* 
                    data['vids'] = torch.FloatTensor(np.ones((1,1)))#don't care for testing
                    data['dir_labels'] = torch.FloatTensor(dir_2) # Direction Label
                    outputs = model.forward(data, rnn_init=rnn_init, mode=opt.mode, step='reconstruct')
                    data['visual_feature'] = outputs['visual_feature']
                    
                    # Create the inputs for the direction prediction module
                    strt_wind_2, end_wind_2 = rel_obj_wind_2 * shift_wind, (rel_obj_wind_2 * shift_wind) + wind_size
                    data['real_audio_mags'] = outputs['pred_spectrogram'][:, :, :, strt_wind_2 : end_wind_2] # Separated spectrogram
                    data['audio_mags'] = outputs['pred_spectrogram'][:, :, :, strt_wind_2 : end_wind_2] # Separated spectrogram
                    # Forward pass through the direction prediction module
                    outputs = model.forward(data, rnn_init=rnn_init, mode=opt.mode, step='dir_pred')
                    # Obtain the direction label
                    pred_label_2 = F.softmax(outputs['pred_dir'], dim=-1).squeeze().argmax().item()
                    if pred_label_2 == dir_2: # If labels match, then update flag
                        corr_flg2 = 1
                    
                    sliding_window_start = sliding_window_start + int(opt.hop_size * opt.audio_sampling_rate)

                #deal with the last segment
                audio_segment = audio_mix[-samples_per_window:]
                audio_mix_mags, audio_mix_phases = generate_spectrogram_magphase(audio_segment, opt.stft_frame, opt.stft_hop) 
                data['audio_mix_mags'] = torch.FloatTensor(audio_mix_mags).unsqueeze(0)
                data['audio_mix_phases'] = torch.FloatTensor(audio_mix_phases).unsqueeze(0)
                data['real_audio_mags'] = data['audio_mix_mags'] #dont' care for testing
                data['audio_mags'] = data['audio_mix_mags'] #dont' care for testing
                #separate for video 1
                data['per_visuals'] = person_visuals_1
                data['edges'] = edges_1
                data['edge_wt'] = edges_attr_1
                data['batch_vec'] = batch_vec_1
                data['labels'] = torch.FloatTensor(np.ones((1,1)))#don't care for testing torch.Tensor([det_box1[1]]).long() #* 
                data['vids'] = torch.FloatTensor(np.ones((1,1)))#don't care for testing
                data['dir_labels'] = torch.FloatTensor(dir_1) # Direction Label
                outputs = model.forward(data, rnn_init=rnn_init, mode=opt.mode, step='reconstruct')
                data['visual_feature'] = outputs['visual_feature']
                
                # Create the inputs for the direction prediction module
                strt_wind_1, end_wind_1 = rel_obj_wind_1 * shift_wind, (rel_obj_wind_1 * shift_wind) + wind_size
                data['real_audio_mags'] = outputs['pred_spectrogram'][:, :, :, strt_wind_1 : end_wind_1] # Separated spectrogram
                data['audio_mags'] = outputs['pred_spectrogram'][:, :, :, strt_wind_1 : end_wind_1] # Separated spectrogram
                # Forward pass through the direction prediction module
                outputs = model.forward(data, rnn_init=rnn_init, mode=opt.mode, step='dir_pred')
                # Obtain the direction label
                pred_label_1 = F.softmax(outputs['pred_dir'], dim=-1).squeeze().argmax().item()
                if pred_label_1 == dir_1: # If labels match, then update flag
                    corr_flg1 = 1
                
                #separate for video 2
                data['per_visuals'] = person_visuals_2
                data['edges'] = edges_2
                data['edge_wt'] = edges_attr_2
                data['batch_vec'] = batch_vec_2
                data['labels'] = torch.FloatTensor(np.ones((1,1)))#don't care for testing torch.Tensor([det_box2[1]]).long() #* 
                data['vids'] = torch.FloatTensor(np.ones((1,1)))#don't care for testing
                data['dir_labels'] = torch.FloatTensor(dir_2) # Direction Label
                outputs = model.forward(data, rnn_init=rnn_init, mode=opt.mode, step='reconstruct')
                data['visual_feature'] = outputs['visual_feature']
                
                # Create the inputs for the direction prediction module
                strt_wind_2, end_wind_2 = rel_obj_wind_2 * shift_wind, (rel_obj_wind_2 * shift_wind) + wind_size
                data['real_audio_mags'] = outputs['pred_spectrogram'][:, :, :, strt_wind_2 : end_wind_2] # Separated spectrogram
                data['audio_mags'] = outputs['pred_spectrogram'][:, :, :, strt_wind_2 : end_wind_2] # Separated spectrogram
                # Forward pass through the direction prediction module
                outputs = model.forward(data, rnn_init=rnn_init, mode=opt.mode, step='dir_pred')
                # Obtain the direction label
                pred_label_2 = F.softmax(outputs['pred_dir'], dim=-1).squeeze().argmax().item()
                if pred_label_2 == dir_2: # If labels match, then update flag
                    corr_flg2 = 1


            # Close the detection files
            obj_det_1.close()
            obj_det_2.close()

            perf_dict[(v1, c1, obj_wind_1, v2, c2, obj_wind_2)] = [corr_flg1, corr_flg2]
            res_dict[(v1, c1, obj_wind_1, v2, c2, obj_wind_2)] = [pred_label_1, dir_1, pred_label_2, dir_2]
            #*print('Previous performance: ' + str(perf_dict)  +  ' Direction Prediction: ' + str(res_dict) )
            #*break # For testing the code
            
    # Over all results
    perf_mat = list(perf_dict.values())
    pred_gt_lab = np.array(list(res_dict.values()))
    #*print('Missed Sequences: ' + str(missed_seq))
    if len(perf_mat) > 0:
        print('Overall nos. correct 1, nos. correct 2: ' + str(np.mean(np.array(perf_mat), axis=0)))
        
        #*"""
        # Save the results
        with open(os.path.join(opt.output_dir_root, 'Fin_Sigma_' + str(opt.sigma) + '_OnlyDirpred_results_noscntxt_' + str(opt.nos_cntxt) + '.pickle'), 'wb') as f:
            pickle.dump([perf_dict, np.mean(np.array(perf_mat), axis=0), res_dict, missed_seq], f)

        #*"""

if __name__ == '__main__':
    main()

import os.path
import librosa
from data.base_dataset import BaseDataset
import h5py
import random
from random import randrange
import glob
import json
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import pickle
import csv

import torch
import torchvision.transforms as transforms

from scipy.sparse import csgraph
import ast

def generate_spectrogram_magphase(audio, stft_frame, stft_hop, with_phase=True):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=stft_frame, center=True)
    spectro_mag, spectro_phase = librosa.core.magphase(spectro)
    spectro_mag = np.expand_dims(spectro_mag, axis=0)
    if with_phase:
        spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
        return spectro_mag, spectro_phase
    else:
        return spectro_mag

def augment_audio(audio):
    audio = audio * (random.random() + 0.5) # 0.5 - 1.5
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def sample_audio(audio, window, sr, obj_win):
    # repeat if audio is too short
    if audio.shape[0] < window:
        n = int(window / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    max_strt = np.int_((audio.shape[0] - window) // sr) + 1
    audio_start = random.choice(np.arange(max_strt)) * sr #* randrange(0, audio.shape[0] - window + 1)
    audio_end = audio_start+window

    # Repeat until the start doesn't include the beginning of the sub_window
    while (audio_start > obj_win * sr) or (audio_end < obj_win * sr): #* or (audio_end < sub_window[1]):
        audio_start = random.choice(np.arange(max_strt)) * sr #* randrange(0, audio.shape[0] - window + 1)
        audio_end = audio_start+window

    audio_sample = audio[audio_start:(audio_start+window)]

    return audio_sample, obj_win - (audio_start // sr) # Return the sampled audio and the new index of the window

def augment_image(image):
	if(random.random() < 0.5):
		image = image.transpose(Image.FLIP_LEFT_RIGHT)
	enhancer = ImageEnhance.Brightness(image)
	image = enhancer.enhance(random.random()*0.6 + 0.7)
	enhancer = ImageEnhance.Color(image)
	image = enhancer.enhance(random.random()*0.6 + 0.7)
	return image

def get_vid_name(npy_path):
    #first 11 chars are the video id
    return os.path.basename(npy_path)[:11]

def get_clip_name(npy_path):
    return os.path.basename(npy_path)[:-4]

def get_frame_root(npy_path):
    frame_path = '/'.join(npy_path.decode('UTF-8').split('/')[:-3])
    return frame_path.encode('UTF-8') #* os.path.join(os.path.dirname(os.path.dirname(npy_path)), 'frame')

def get_audio_root(npy_path):
    audio_root = '../MUSIC_Dataset/Extracted_Audio/'
    audio_path = '/'.join(npy_path.decode('UTF-8').split('/')[-5:-3])
    return os.path.join(audio_root, audio_path).encode('UTF-8') #* os.path.join(os.path.dirname(os.path.dirname(npy_path)), 'audio_11025')

def sample_object_detections(detection_bbs):
    class_index_clusters, aud_vis_obj = {}, {} #get the indexes of the detections for each class
    for i in range(detection_bbs.shape[0]):
        if int(detection_bbs[i,0]) in class_index_clusters.keys():
            class_index_clusters[int(detection_bbs[i,0])].append(i)
            aud_vis_obj[int(detection_bbs[i,0])].append(int(detection_bbs[i,1]))
        else:
            class_index_clusters[int(detection_bbs[i,0])] = [i]
            aud_vis_obj[int(detection_bbs[i,0])] = [int(detection_bbs[i,1])]

    detection2return, idx_lst = np.array([]), []
    for cls in class_index_clusters.keys():
        sampledIndex = np.array(class_index_clusters[cls]) 
        idx_lst.append(detection_bbs[sampledIndex,2:]) # Store the index of the bounding box
        if detection2return.shape[0] == 0:
            detection2return = detection_bbs[sampledIndex,:]
        else:
            detection2return = np.concatenate((detection2return, detection_bbs[sampledIndex,:]), axis=0) # Constructing a 2d array
    return detection2return, aud_vis_obj 

def compute_overlap(bbox_inst, bbox_plr):
    """ Computes the overlap between instrument and player bounding boxes and returns the IoU of overlapping pixels """
    x1min, y1min, x1max, y1max = bbox_inst
    x2min, y2min, x2max, y2max = bbox_plr
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max) 
    ymax = min(y1max, y2max)
    iw = max(xmax - xmin, 0)
    ih = max(ymax - ymin, 0)
    inters = iw * ih
    union = (x1max - x1min) * (y1max - y1min) + (x2max - x2min) * (y2max - y2min) - inters
    return inters * 1. / union

class ASIW(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.NUM_PER_MIX = opt.num_per_mix
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.audio_window = opt.audio_window
        random.seed(opt.seed)
        self.iou_thresh = 0.1
        self.conf_thresh = 0.4
        self.feature_dim = 2048
        self.sigma = opt.sigma
        self.num_object_per_video = opt.num_object_per_video # Maximum number of objects present in a video
        # Class labels for the different octants
        if self.opt.number_of_dir_classes == 8:
            self.disp_cls = {str([1,1,1]): 0, str([-1,1,1]): 1, str([-1,-1,1]): 2, str([1,-1,1]): 3, \
                str([1,1,-1]): 4, str([-1,1,-1]): 5, str([-1,-1,-1]): 6, str([1,-1,-1]): 7}
        elif self.opt.number_of_dir_classes == 10:
            self.disp_cls = {str([0,0,0]): 0, str([1,1,1]): 1, str([-1,1,1]): 2, str([-1,-1,1]): 3, str([1,-1,1]): 4, \
                str([1,1,-1]): 5, str([-1,1,-1]): 6, str([-1,-1,-1]): 7, str([1,-1,-1]): 8}
        else:
            self.disp_cls = {str([0,0,0]): 0, str([0,0,1]): 1, str([0,1,0]): 2, str([1,0,0]): 3, str([0,0,-1]): 4, str([0,-1,0]): 5, str([-1,0,0]): 6, \
                str([0,1,1]): 7, str([1,1,0]): 8, str([1,0,1]): 9, str([0,1,-1]): 10, str([1,-1,0]): 11, str([1,0,-1]): 12, str([-1,1,0]): 13, str([-1,0,1]): 14, \
                str([0,-1,1]): 15, str([0,-1,-1]): 16, str([-1,-1,0]): 17, str([-1,0,-1]): 18, str([1,1,1]): 19, str([-1,1,1]): 20, str([-1,-1,1]): 21, \
                str([1,-1,1]): 22, str([1,1,-1]): 23, str([-1,1,-1]): 24, str([-1,-1,-1]): 25, str([1,-1,-1]): 26}
        #initialization
        self.detection_dic = {} #gather the clips for each video
        # Define the thresholds for considering no displacement
        self.u_thresh, self.v_thresh, self.d_thresh = 0.60, 0.60, 0.30
        #load the list of valid videos
        file_list_path = os.path.join(opt.hdf5_path, 'Multiple_Valid_RAFTFlow_Seq.pickle') #* Multiple_Valid_RAFTFlow_Seq_New.pickle #* 'Single_Valid_RAFTFlow_Seq.pickle' 'Valid_Dir_Videos_Vis_Text.pickle'
        # Read in the file names
        with open(file_list_path, 'rb') as f:
            train_vid_lst, val_vid_lst, test_vid_lst = pickle.load(f)
        # Assign the file names
        if opt.mode == 'train': 
            detections = train_vid_lst
            self.mode = opt.mode
        else: 
            detections = test_vid_lst
            self.mode = 'test'
        # Initialize the root
        self.data_path = opt.data_path

        for detection in detections:
            vidname = detection #get video id
            if vidname in self.detection_dic.keys():
                self.detection_dic[vidname].append(os.path.join(self.data_path, 'Extracted_Frames', self.mode, vidname, 'Vision_Text_Labels.csv'))
            else:
                self.detection_dic[vidname] = [os.path.join(self.data_path, 'Extracted_Frames', self.mode, vidname, 'Vision_Text_Labels.csv')]

        if opt.mode != 'train':
            vision_transform_list = [transforms.Resize((224,224)), transforms.ToTensor()]
            self.videos2Mix = [random.sample(self.detection_dic.keys(), self.NUM_PER_MIX) for _ in range(self.opt.batchSize * self.opt.validation_batches)]
        elif opt.preserve_ratio:
            vision_transform_list = [transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor()]
        else:
            vision_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor()]
        if opt.subtract_mean:
            vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.vision_transform = transforms.Compose(vision_transform_list)

        #load hdf5 file of scene images
        if opt.with_additional_scene_image:
            h5f_path = os.path.join(opt.scene_path)
            h5f = h5py.File(h5f_path, 'r')
            self.scene_images = h5f['image'][:]


    def __getitem__(self, index):
        if self.mode == 'val': # In order to validate on the same samples everytime
            videos2Mix = self.videos2Mix[index]
        else:
            videos2Mix = random.sample(self.detection_dic.keys(), self.NUM_PER_MIX) #get videos to mix
        clip_det_paths = [None for n in range(self.NUM_PER_MIX)]
        clip_det_bbs = [None for n in range(self.NUM_PER_MIX)]
        idx_lst = [None for n in range(self.NUM_PER_MIX)]
        disp_lst = [None for n in range(self.NUM_PER_MIX)]
        img_obj_wind = {0: "0001.png", 1: "0009.png", 2: "0017.png", 3: "0025.png", 4: "0033.png", 5: "0041.png", 6: "0049.png", 7: "0057.png", 8: "0065.png", 9: "0073.png"}
        for n in range(self.NUM_PER_MIX):
            if self.mode == 'val':
                clip_det_paths[n] = (self.detection_dic[videos2Mix[n]]) 
            else:
                clip_det_paths[n] = random.choice(self.detection_dic[videos2Mix[n]]) 

            # Load the displacement file for the sample
            with open(os.path.join(self.data_path, 'Rectified_Frames', self.mode, clip_det_paths[n].split('/')[-2], 'Comb_RAFTFlow_Displacement_Vectors_New.pickle'), 'rb') as f:
                disp = pickle.load(f)

            # Augment to the list of displacements
            disp_lst[n] = disp
            
            #* Load the detection file
            with open(clip_det_paths[n], 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                # Initialize the bbox file
                det_bbox = []
                for r in csv_reader:
                    det_bbox.append(r)
            # Sample the relevant object bounding boxes - Not necessary now, since we are loading object info from the dict
            clip_det_bbs[n], idx_lst[n] = sample_object_detections(np.array(det_bbox)) #load the bbs for the clip and sample one from each class

        audios, audios_z = [], [] #audios of mixed videos
        
        ind_v, grph_cnt = [], 0 
        objects_visuals = []
        
        person_visuals = []
        objects_labels = []
        objects_disp = [] 
        objects_audio_mag = []
        objects_wind = [] 
        objects_audio_phase = []
        
        objects_vids = []
        wt_ind, dir_wt_ind = [], []
        objects_real_audio_mag = [] 
        objects_audio_mix_mag = []
        
        objects_audio_mix_phase = []
        
        tot_nodes, edge_wt, sub, obj = 0, [], [], []

        for n in range(self.NUM_PER_MIX): # Iterate over each video to mix
            val_nodes, val_nodes_nm, new_objects_visuals, new_person_visuals, new_ind_v, new_sub, new_obj, obj_per_vid = [], [], [], [], [], [], [], 0
            rbf_flg, vid = 0, random.randint(1,100000000000) #generate a unique video id
            #
            audio_path = os.path.join(self.data_path, 'Extracted_Audio', self.mode, clip_det_paths[n].split('/')[-2] + '.aac')
            audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate) #* [2:-1]
            
            detection_bbs = clip_det_bbs[n]
            
            # Extract clip name
            vid_nm = clip_det_paths[n].split('/')[-2] # Get the full vid
            # Load the json file
            with open(os.path.join('/'.join(clip_det_paths[n].split('/')[:-1]), 'Image_info_20obj.json'), 'r') as f:
                vid_meta = json.load(f)
            # Open the h5py file for man/woman node
            obj_det = h5py.File(os.path.join('/'.join(clip_det_paths[n].split('/')[:-1]), 'Image_feature_20obj.h5'), 'r')
            # Determine the list of objects for the current video
            obj_lst = list(disp_lst[n].keys())
            # Obtain valid auditory classes for this video
            aud_cls_lst = list(idx_lst[n].keys())
            # Load the pre-computed Chamfer distance for this video
            with open(os.path.join(self.data_path, 'Rectified_Frames', self.mode, clip_det_paths[n].split('/')[-2], 'Chamfer_Info.pickle'), 'rb') as f: #* _New
                vid_info = pickle.load(f)
            
            # Define the list of frame and bounding box information
            obj_con, p_i = {}, 0
            for o_c in vid_info['object_context'][:-1]: # Loop over all nodes except the with_additional_image node
                if (o_c[0] == -1) and (o_c[1] == -1): continue # Skip the with_additional_image node
                p_o_ind = o_c[2]
                if p_o_ind == 1: # If it is a principal object
                    obj_con[vid_info['object_wind'][p_i]] = [] # Initialize the list
                    p_i += 1

            p_i = 0 # Reinitialize the principal object counter
            for o_c in vid_info['object_context'][:-1]:
                if (o_c[0] == -1) and (o_c[1] == -1): continue # Skip the with_additional_image node
                p_o_ind = o_c[2]
                if p_o_ind == 1: # If it is a principal object
                    p_i += 1            
                else: # If it is a context object
                    obj_con[vid_info['object_wind'][p_i]].append(o_c)
            
            # Load the Chamfer distance matrix
            chamf_dist = vid_info['dist_mat'] #*np.array(list(vid_info['dist_mat'].values()))
            
            # Design appropriate edge matrix
            e_w = chamf_dist.reshape(-1)
            e_w = e_w[~ np.isnan(e_w)]
            if e_w.shape[0] > 0:
                max_d = e_w.max()
            else:
                max_d = 0

            # Filter out the non-zero elements from the edge-weight matrix
            nz_e_w = e_w[np.nonzero(e_w)].reshape(-1) 
            # Check if sufficient number of non-zero elements exist
            if (max_d > 0) and (nz_e_w.shape[0] > 0):
                med_wt = np.percentile(nz_e_w, self.opt.sigma) #* np.median(e_w[np.nonzero(e_w)]) # This returns the median value of the non-zero elements of the edge weights
                if med_wt == 0:
                    med_wt = 1.0
                    #*print('Smoothing error!')
                rbf_flg = 1 # Compute RBF kernel for this video
            else:
                med_wt = 0
                rbf_flg = 0 # No RBF kernel for this video
                #*print('All zeros for: ' + str(clip_det_paths[n].split('/')[-2]))

            # Only keep num_object_per_video sound sources
            for i in range(min(len(vid_info['object_wind']), self.num_object_per_video)): # Iterate over each source detected
                # Choose a window for the current object - randomly
                vis_obj, obj_wind = vid_info['object_wind'][i] 
                # Check if the current object has been chosen from the csv file
                if str(vis_obj) in detection_bbs[:,1]: # This check becomes trivial now
                    obj_idx = list(detection_bbs[:,1]).index(str(vis_obj)) # Index of chosen object in csv file

                    audio_segment, rel_obj_wind = sample_audio(audio, self.audio_window, self.opt.audio_sampling_rate, obj_wind)
                    if(self.opt.enable_data_augmentation and self.mode == 'train'):
                        audio_segment = augment_audio(audio_segment)

                    # Make a copy for the mixture only for true audio snippets
                    audios.append(audio_segment)
                    objects_wind.append(rel_obj_wind)
                    # Convert this chosen window to frequency domain
                    audio_mag, audio_phase = generate_spectrogram_magphase(audio_segment, self.stft_frame, self.stft_hop)
                    objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
                    objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))

                    label = int(detection_bbs[obj_idx,0]) + 1 # Since label 0 is reserved for backgorund/extra label
                    
                    img_name = str(img_obj_wind[obj_wind]) # Extract visual info from first frame of selected window 
                    im_id = [i_ for i_ in range(len(vid_meta)) if vid_meta[i_]['image'] == img_name][0]  # Get the im_id for the matched frame
                    # Since displacement pickle contains current window, so object must be present in first frame
                    obj_frm_idx = vid_meta[im_id]['objects'].index(int(vis_obj)) # Find the index of the sounding object from the list of objects in the first frame of the window
                    
                    # Count the number of objects
                    obj_per_vid += 1

                    # Get the list of context nodes
                    con_lst = obj_con[(vis_obj, obj_wind)]
                   
                    # Extract the context information for the current 
                    if len(con_lst) > 0:
                        # If atleast one context node is available, then include all valid context nodes
                        for o_c in con_lst:
                            c_im_id, p, _ = o_c
                            new_person_visuals.append(torch.from_numpy(obj_det['features'][c_im_id, p, :])) # Obtain the context node features
                            new_ind_v.extend([grph_cnt+1])
                            val_nodes.extend([tot_nodes])
                            val_nodes_nm.extend([(c_im_id, p)])
                            tot_nodes += 1
                    #*else:
                    # Incorporate the principal object information
                    new_person_visuals.append(torch.from_numpy(obj_det['features'][im_id, obj_frm_idx, :])) # Obtain the sound node features
                    objects_labels.append(label)

                    objects_vids.append(vid)
                    #*per_ind[n].append(0) # Indicator if context node exists
                    new_ind_v.extend([grph_cnt+1]) # Context node should not be part of the graph
                    # Determine the displacement of the object
                    disp_vec = disp_lst[n][vis_obj][obj_wind]
                    # Check if the vector should go to class 0, if not then normalize
                    if (np.abs(disp_vec[0]) >= self.u_thresh) or (np.abs(disp_vec[1]) >= self.v_thresh) or (np.abs(disp_vec[2]) >= self.d_thresh): 
                            # Check if 10-class setting
                            if self.opt.number_of_dir_classes == 10:
                                # Replace 0s along a dimension (if any)
                                disp_vec = np.where(disp_vec == 0, np.ones_like(disp_vec), disp_vec)
                                # Convert it to a unit vectors along each dimension
                                disp_vec /= np.absolute(disp_vec)
                                # Covert to displacement vector
                                disp_vec = list(np.int_(disp_vec))
                                # Introduce the label
                                objects_disp.append(self.disp_cls[str(disp_vec)])
                            elif self.opt.number_of_dir_classes == 28: # For 28 class setting
                                basis_vecs_str, basis_vecs = list(self.disp_cls.keys()), []
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
                                objects_disp.append(self.disp_cls[u_v])
                    else:
                            # Implies there is no motion
                            objects_disp.append(self.disp_cls[str([0,0,0])])
                    # Incorporate edge information - only self loop here
                    val_nodes.extend([tot_nodes]) #*sub.extend([tot_nodes + 1])
                    val_nodes_nm.extend([(im_id, obj_frm_idx)])
                    tot_nodes += 1 # Always increment by 1, since there is only an object node
                    

                    wt_ind.append(1) # An indicator indicating if loss for this sample will be computed
                    dir_wt_ind.append(1) # An indicator indicating if direction loss for this sample will be computed

            # Check if maximum nos. of objects covered, if not introduce audio input/gt and visual feature nodes for the difference
            if obj_per_vid < self.num_object_per_video:
                nos_missing_obj = self.num_object_per_video - obj_per_vid # Store the difference count
                # Add audio signal for the missing elements
                for _ in range(nos_missing_obj):
                    #*new_objects_visuals.append(self.vision_transform(object_image).unsqueeze(0)) # Will be zeroed out eventually
                    new_person_visuals.append(torch.zeros(1, self.opt.feat_dim).float()) # Zero sound node in this case
                    
                    audio_segment, rel_obj_wind = sample_audio(audio, self.audio_window, self.opt.audio_sampling_rate, obj_win = 0)
                    if(self.opt.enable_data_augmentation and self.mode == 'train'):
                        audio_segment = augment_audio(audio_segment)
                    # Convert this chosen window to frequency domain
                    audio_mag, audio_phase = generate_spectrogram_magphase(audio_segment, self.stft_frame, self.stft_hop)
                    # Augment the audio for the last true sample
                    objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
                    objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
                    objects_vids.append(vid)
                    objects_labels.append(0)
                    new_ind_v.extend([0]) # Context node should not be part of the graph
                    objects_wind.append(rel_obj_wind)
                    # Introduce the label
                    if self.opt.number_of_dir_classes == 8:
                        objects_disp.append(self.disp_cls[str([1,1,1])])
                    else:
                        objects_disp.append(self.opt.number_of_dir_classes - 1) # Final class for background/extra
                    # Increment tot_nodes
                    tot_nodes += 1 # Always increment by 2, since there are 2 nodes both (zero/zero) person node and an object node
                    wt_ind.append(0) # An indicator indicating if loss for this sample will be computed
                    dir_wt_ind.append(0) # An indicator indicating if direction loss for this sample will be computed

            #add an additional scene image for each video
            if self.opt.with_additional_scene_image and (vid in objects_vids): #* or (vid in objects_vids_z)
                new_person_visuals.append(torch.randn_like(torch.from_numpy(obj_det['features'][0, 0, :]))) # Choose the first box of the first im_id as the random object
                objects_labels.append(0) #* self.opt.number_of_classes - 1
                
                # Make a copy for the mixture only if no true source found
                audios_z.append(audio_segment)
                # Augment the audio for the last true sample
                objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
                objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
                objects_vids.append(vid)
                new_ind_v.extend([grph_cnt+1]) # First element to be excluded eventually from graph
                objects_wind.append(rel_obj_wind)
                # Introduce the label
                if self.opt.number_of_dir_classes == 8:
                    objects_disp.append(self.disp_cls[str([1,1,1])])
                else:
                    objects_disp.append(self.opt.number_of_dir_classes - 1) # Final class for background/extra

                # Incorporate edge information - only self loop here
                val_nodes.extend([tot_nodes]) #*sub.extend([tot_nodes + 1])
                val_nodes_nm.extend([(-1, -1)])
                #*obj.extend([tot_nodes + 1])
                wt_ind.append(1)
                dir_wt_ind.append(0) # An indicator indicating if direction loss for this sample will be computed
                # Increment tot_nodes
                tot_nodes += 1 # Always increment by 1, since there is only an object node

            grph_cnt += 1
            obj_det.close()

            # Define the edges for this graph 
            for v1_c, (v1, v1_idx) in enumerate(zip(val_nodes_nm, val_nodes)): # Bidirectional, each graph has 2 nodes -- object + person
                for v2_c, (v2, v2_idx) in enumerate(zip(val_nodes_nm, val_nodes)):
                    # Check if rbf kernel is to be computed
                    if rbf_flg == 1: # Check if Chamfer distance matrix is the same shape as current node
                        if v1_c < v2_c: # Upper traiangular matrix part
                            edge_wt.append(np.exp((vid_info['dist_mat'][(v1_c, v2_c)]/max_d) * -1./ med_wt)) #* /max_d
                        elif v2_c < v1_c: # Lower traiangular matrix part
                            edge_wt.append(np.exp((vid_info['dist_mat'][(v2_c, v1_c)]/max_d) * -1./ med_wt)) #* /max_d
                        else: # For self loops
                            edge_wt.append(1)
                    else:
                        edge_wt.append(1)

                    new_sub.append(v1_idx)
                    new_obj.append(v2_idx)

            # Augment the batch_vec indicator
            ind_v.extend(new_ind_v)
            
            person_visuals.extend(new_person_visuals)
            sub.extend(new_sub)
            obj.extend(new_obj)

        #mix audio and make a copy of mixed audio spec for each object
        if len(audios) > 0: # Atleast 1 valid object has been detected for the pair of videos
            audio_mix = np.asarray(audios).sum(axis=0) / len(audios) #* self.NUM_PER_MIX
            audio_mix_mag, audio_mix_phase = generate_spectrogram_magphase(audio_mix, self.stft_frame, self.stft_hop)
        else:
            audio_mix = np.asarray(audios_z).sum(axis=0) / len(audios_z) #* self.NUM_PER_MIX
            audio_mix_mag, audio_mix_phase = generate_spectrogram_magphase(audio_mix, self.stft_frame, self.stft_hop) # Random assignment
        
        for n in range(self.NUM_PER_MIX):
            for i in range(self.num_object_per_video): 
                objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mag).unsqueeze(0))
                objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phase).unsqueeze(0))

            if self.opt.with_additional_scene_image:
                objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mag).unsqueeze(0))
                objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phase).unsqueeze(0))

        
        audio_mags = np.vstack(objects_audio_mag) #audio spectrogram magnitude
        
        audio_phases = np.vstack(objects_audio_phase) # audio spectrogram phase
        
        labels = np.vstack(objects_labels) # labels for each object, 0 denotes padded object

        windows = np.vstack(objects_wind) # relative window indices for each video
        
        vids = np.vstack(objects_vids) # video indexes for each object, each video should have a unique id
        
        audio_mix_mags = np.vstack(objects_audio_mix_mag) 
        
        audio_mix_phases = np.vstack(objects_audio_mix_phase) 
        
        edges = torch.Tensor([sub, obj]).long()
        #*print('Shape of edges: ' + str(edges.shape) + ' edges: ' + str(edges))

        dir_labels = np.vstack(objects_disp) # labels for the direction of object motion

        data = {'labels': labels, 'dir_labels': dir_labels, 'audio_mags': audio_mags, 'audio_mix_mags': audio_mix_mags, 'vids': vids}
        data['windows'] = windows
        data['edges'] = edges
        data['edge_wt'] = torch.Tensor(edge_wt)
        data['batch_vec'] = ind_v # Of List type 
        data['loss_ind'] = torch.Tensor(wt_ind)
        data['dir_loss_ind'] = torch.Tensor(dir_wt_ind)
        
        if self.opt.mode == 'val' or self.opt.mode == 'test':
            data['audio_phases'] = audio_phases
            #*data['audio_phases_z'] = audio_phases_z
            data['audio_mix_phases'] = audio_mix_phases
            #*data['audio_mix_phases_z'] = audio_mix_phases_z
        
        data['per_visuals'] = np.vstack(person_visuals)
        #*print('Loss indicator: ' + str(data['loss_ind']) + ' Direction prediction indicator: ' + str(data['dir_loss_ind']))
        return data

    def __len__(self):
        if self.opt.mode == 'train':
            return self.opt.batchSize * self.opt.num_batch
        elif self.opt.mode == 'val':
            return self.opt.batchSize * self.opt.validation_batches

    def name(self):
        return 'ASIW'

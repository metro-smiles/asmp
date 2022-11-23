import numpy as np
import torch
import os
from torch import optim
import torch.nn.functional as F
from . import networks, criterion
from utils.utils import warpgrid
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        self.window_size, self.overlap_size = 46, 4
        #initialize model and criterions
        self.net_visual, self.net_unet, self.net_classifier, self.graph_nets, self.map_net, self.rnn, self.rnn_classif, self.net_fusion_enc, self.net_fuse = nets

    def forward(self, input, rnn_init, mode='train', step='', opt=None):
        labels = input['labels'].squeeze().long() 
        vids = input['vids'] 
        batch_vec = input['batch_vec'] 
        audio_mags =  input['audio_mags'] 
        audio_mix_mags = input['audio_mix_mags'] 
        audio_mix_mags = audio_mix_mags + 1e-10

        persons = input['per_visuals']
        
        if mode != 'test': # If in training mode or val mode
            edge = input['edges']
            edge_wt = input['edge_wt'].view(-1) 
            
            if not self.opt.only_separation:
                dir_labels = input['dir_labels'].squeeze().long()
                wind = input['windows'].squeeze().long()

            # If there is atleast one node in the batch for which graph processing needs to be done
            if edge.size(1) > 0: 
                
                person_feature = self.map_net(persons) #* Nos. of (person) objects x 2048 -> Nos. of (person) objects x 512
                batch_vec = input['batch_vec'] 

                x = person_feature.view(-1, person_feature.size(1)) # Interleaved tensor with person and instrument nodes

                # Forward through the graph embedding network
                if torch.min(batch_vec) <= 0:
                    graph_embed = self.graph_nets((x, edge, edge_wt, batch_vec))[1:, :] #* Ignore the 0-person features
                else:
                    batch_vec = batch_vec - 1
                    graph_embed = self.graph_nets((x, edge, edge_wt, batch_vec))
                
                # Initialize the squared cosine similarity
                dot_p2 = 0
                num_object_per_video = self.opt.num_object_per_video
                if self.opt.with_additional_scene_image:
                    num_object_per_video += 1
                # Run the recurrence through the RNN
                for t in range(num_object_per_video): 
                    # Forward pass through the RNN
                    if t == 0: # For the first object
                        g_e, rnn_init = self.rnn(graph_embed.unsqueeze(0), rnn_init)
                        g_e = g_e.squeeze()
                        # Normalize the RNN Output
                        cl = g_e / (g_e.norm(dim=1) + 1e-10)[:, None] 
                        cl = cl.unsqueeze(1)
                    else:
                        tmp, rnn_init = self.rnn(graph_embed.unsqueeze(0), rnn_init)
                        tmp = tmp.squeeze()
                        # Augment it to the previous embeddings
                        g_e = torch.cat([g_e, tmp], dim=1)
                        # Normalize the RNN Output
                        tmp = (tmp / (tmp.norm(dim=1) + 1e-10)[:, None]).unsqueeze(1)
                        # Compute the squared cosine similarity
                        dot_p2 += (torch.bmm(cl, tmp.transpose(1, 2)).mean())**2
                        # Augment the current RNN output to the ones from the previous time steps
                        cl = torch.cat([cl, tmp], dim=1)
                        
                    
                # Compute the feature for U-Net
                graph_embed = g_e.view(-1, person_feature.size(1))

                visual_feature = graph_embed.unsqueeze(2).unsqueeze(2) # Extract out only the attended instrument features #* [:, person_feature.size(1):]
                
            else:
                visual_feature = persons #* self.net_visual(Variable(persons, requires_grad=False)) #* self.map_net()

            # warp the spectrogram
            B = audio_mix_mags.size(0)
            T = audio_mix_mags.size(3) #* 256
            #*print('Shape of mix tensor: ' + str(audio_mix_mags.shape))
            if self.opt.log_freq:
                grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).to(self.opt.device)
                audio_mix_mags = F.grid_sample(audio_mix_mags, grid_warp)
                audio_mags = F.grid_sample(audio_mags, grid_warp)

            # calculate ground-truth masks
            gt_masks = audio_mags / audio_mix_mags
            # clamp to avoid large numbers in ratio masks
            gt_masks.clamp_(0., 5.)

            #*print('Shape of audio feature, post warp: ' + str(audio_mix_mags.size()))
            # audio-visual feature fusion through UNet and predict mask
            audio_log_mags = torch.log(audio_mix_mags).detach() #*; print('Shape of audio_log_mags: ' + str(audio_log_mags.size()));
            mask_prediction = self.net_unet(audio_log_mags, visual_feature) #*) torch.randn(26, 512, 1, 1).cuda()

            # masking the spectrogram of mixed audio to perform separation
            separated_spectrogram = (audio_mix_mags * mask_prediction) #*.detach()
            #*print('Shape of separated_spectrogram: ' + str(separated_spectrogram.shape))
            # generate spectrogram for the classifier
            spectrogram2classify = torch.log(separated_spectrogram + 1e-10) #get log spectrogram

            # calculate loss weighting coefficient
            if self.opt.weighted_loss:
                weight = torch.log1p(audio_mix_mags)
                weight = torch.clamp(weight, 1e-3, 10)
            else:
                weight = None

            #classify the predicted spectrogram
            label_prediction = self.net_classifier(spectrogram2classify)

            # If only performing separation
            if self.opt.only_separation:
                output = {'gt_label': labels, 'pred_label': label_prediction, 'pred_mask': mask_prediction, 'gt_mask': gt_masks, \
                        'rnn_labels': None, 'rnn_cls': dot_p2, \
                        'pred_spectrogram': separated_spectrogram, 'visual_object': None, 'audio_mix_mags': audio_mix_mags, 'weight': weight, 'vids': vids} #* labels.view(-1).long()
            else:                
                # Fusion Modules
                shift_window = self.window_size - self.overlap_size # The number of columns by which the window shifts
                # Define the appropriate indices
                strt_wind = (wind[:] * shift_window) #*- 1
                end_wind = (wind[:] * shift_window) + self.window_size

                # Select the relevant sub-spectrogram
                sub_spec = torch.cat([spectrogram2classify[_, :, :, strt_wind[_]:end_wind[_]].unsqueeze(0) for _ in range(spectrogram2classify.size(0))], dim=0) 
                
                # Encode the spectrogram 
                spec_enc = self.net_fusion_enc(sub_spec)
                
                dir_prediction = self.net_fuse(spec_enc)

                output = {'gt_label': labels, 'pred_label': label_prediction, 'pred_mask': mask_prediction, 'gt_mask': gt_masks, \
                        'rnn_labels': None, 'rnn_cls': dot_p2, 'dir_labels': dir_labels, 'pred_dir': dir_prediction, \
                        'pred_spectrogram': separated_spectrogram, 'visual_object': None, 'audio_mix_mags': audio_mix_mags, 'weight': weight, 'vids': vids} #* labels.view(-1).long()
        else: # If in test mode
            if step == 'reconstruct':
                #*labels = labels.squeeze().long() #covert back to longtensor
                person_feature = self.map_net(persons)
                person_feature = person_feature.view(-1, person_feature.size(1)) #* Nos. of (person) objects x 2048 -> Nos. of (person) objects x 512 
                batch_vec = input['batch_vec'] #*.to(self.opt.device) #*input['node_list_obj']
                edge = input['edges']
                edge_wt = input['edge_wt'].view(-1)
                # If there is atleast one person detected in the batch
                if edge != None: 

                    graph_ip = person_feature 
                    
                    visual_feature = self.graph_nets((graph_ip, edge, edge_wt, batch_vec)) 
                    
                    visual_feature = visual_feature.view(1, -1) 

                else:
                    visual_feature = person_feature 

                # Obtain the RNN output for this step
                g_e, _ = self.rnn(visual_feature.unsqueeze(0), rnn_init)
                visual_feature = g_e.squeeze(0)

                # warp the spectrogram
                B = audio_mix_mags.size(0)
                T = audio_mix_mags.size(3) 
                if self.opt.log_freq:
                    grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).to(self.opt.device)
                    audio_mix_mags = F.grid_sample(audio_mix_mags, grid_warp)
                    audio_mags = F.grid_sample(audio_mags, grid_warp)

                # calculate ground-truth masks
                gt_masks = audio_mags / audio_mix_mags
                # clamp to avoid large numbers in ratio masks
                gt_masks.clamp_(0., 5.)

                # audio-visual feature fusion through UNet and predict mask
                audio_log_mags = torch.log(audio_mix_mags).detach() 

                mask_prediction = self.net_unet(audio_log_mags, visual_feature.unsqueeze(2).unsqueeze(2)) 

                # masking the spectrogram of mixed audio to perform separation
                separated_spectrogram = audio_mix_mags * mask_prediction

                # calculate loss weighting coefficient
                if self.opt.weighted_loss:
                    weight = torch.log1p(audio_mix_mags)
                    weight = torch.clamp(weight, 1e-3, 10)
                else:
                    weight = None
                
                output = {'gt_label': labels, 'pred_label': None, 'pred_mask': mask_prediction, 'visual_feature': visual_feature, \
                         'pred_spectrogram': separated_spectrogram, 'visual_object': None, 'audio_mix_mags': audio_mix_mags, 'weight': weight, 'vids': vids}
            
            elif step == 'dir_pred':

                dir_labels = input['dir_labels'].squeeze().long()
                spectrogram2classify = torch.log(audio_mags + 1e-10) 
                # Encode the spectrogram 
                spec_enc = self.net_fusion_enc(spectrogram2classify)
                dir_prediction = self.net_fuse(spec_enc)

                output = {'gt_label': labels, 'dir_labels': dir_labels, 'pred_dir': dir_prediction,  \
                         'pred_spectrogram': None, 'visual_object': None, 'audio_mix_mags': audio_mix_mags, 'weight': None, 'vids': vids}

        return output


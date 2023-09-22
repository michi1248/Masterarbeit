import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tqdm
from PIL import Image
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from helpers import *


#TODO funtion die mu signal resamples wenn nicht von marius gegeben
#TODO funktion die mir sagt welcher channel ich nehmen will wenn arrayindex sage
#TODO kommunikation mit EMG in echtzeit




class Heatmap:
    def __init__(self, movement_name,path_to_subject_dat,sampling_frequency=2048,path_to_save_plots=r"D:\Lab\MasterArbeit\plots\Heatmaps",frame_duration=150):
        """

        :param movement_name:
        :param path_to_subject_dat:
        :param sampling_frequency:
        :param path_to_save_plots:
        :param frame_duration: (in ms) we want to observe the heatmap before next update
        num_samples = number of frames we have to take in order to update the figure every frame_duration ms
        number_observation_samples = number of samples between two frames
        self.last_frame = last frame is the sample number of the last frame, important to calculate the firings since last frame
        self.closest_mu = closest_mu is the MU that fired the closest to the last frame, important if no mu fires since the last frame to use the old one again
        self.global_counter = global_counter is the number of frames that have been plotted, important because for first time we need to plot colorbar and other times not
        """
        self.movement_name = movement_name
        if not os.path.exists(os.path.join(path_to_save_plots, movement_name)):
            os.makedirs(os.path.join(path_to_save_plots, movement_name))
        self.path_to_save_plots = os.path.join(path_to_save_plots, movement_name)
        self.emg_data,self.Mu_data,self.ref_data = open_all_files_for_one_patient_and_movement(path_to_subject_dat, movement_name)
        self.emg_data, self.mask_for_heatmap = fill_empty_array_indexes_with_0(self.emg_data)
        self.ref_data = self.ref_data.to_numpy()
        self.sampling_frequency = sampling_frequency
        self.fig, (self.ax_emg, self.ax_MU, self.ax_ref) = plt.subplots(3, 1, figsize=(8, 10))
        self.sample_length = self.emg_data[0][1][0].shape[0]
        self.num_samples = int(self.sample_length/(self.sampling_frequency*(frame_duration/1000)))
        self.frame_duration = frame_duration
        self.number_observation_samples = int((self.frame_duration / 1000) * self.sampling_frequency)

        print("calculating MUAPs with STA ...")
        self.muaps = spikeTriggeredAveraging(self.emg_data, self.Mu_data, 0.05, self.sample_length, self.sampling_frequency)
        print("done")
        self.last_frame = 0
        self.closest_mu = 0



    def make_heatmap_emg(self,frame):
        """
        :return: heatmap of emg data

        """
        heatmap = calculate_emg_rms_grid(self.emg_data, frame, self.number_observation_samples)
        normalized_heatmap = normalize_2D_array(heatmap)

        #only do the following if +- window size near extrema
        if (is_near_extremum(frame,self.local_maxima,self.local_minima,time_window=self.frame_duration, sampling_frequency=self.sampling_frequency)):
            # find out to which part of the movement the current sample belongs (half of flex or ref)
            belongs_to_movement,distance = check_to_which_movement_cycle_sample_belongs(frame,self.local_maxima,self.local_minima)
            # add the heatmap to the list of all heatmaps of the fitting flex/ex for later calculation of difference heatmap
            if (belongs_to_movement == 1):
                # the closer the sample is to the extrema the more impact it has on the heatmap
                self.heatmaps_flex = np.add(self.heatmaps_flex, np.multiply(normalized_heatmap, 1/(distance+0.1) ))
                # add heatmap and not normalized heatmap because impact of all heatmaps will be the same but maybe some heatmaps have higher values and i want to use this ??
                #TODO maybe change this (better to use normalized or not ?)
                #IT IS BETER TO USE NORMALIZED BECAUSE IF EMG SCHWANKUNGEN
                self.number_heatmaps_flex +=1
            else:
                self.heatmaps_ex = np.add(self.heatmaps_ex,  np.multiply(normalized_heatmap, 1/(distance+0.1) ))
                self.number_heatmaps_ex += 1

        return normalized_heatmap


    def animate(self):
        num_rows, num_cols = self.emg_data.shape
        self.heatmaps_flex  = np.zeros((num_rows, num_cols))
        self.heatmaps_ex = np.zeros((num_rows, num_cols))
        self.number_heatmaps_flex = 0
        self.number_heatmaps_ex = 0
        self.local_maxima,self.local_minima = get_locations_of_all_maxima(self.ref_data[:])


    def channel_extraction(self):
        """
        :return: List channels_flexion + List channels extension, each entrie in lists is [row,col] of channel
        channels that are most likely to be used for flexion and extension
        """
        mean_flex_heatmap = normalize_2D_array(np.divide(self.heatmaps_flex, self.number_heatmaps_flex))
        mean_ex_heatmap = normalize_2D_array(np.divide(self.heatmaps_ex, self.number_heatmaps_ex))
        difference_heatmap = normalize_2D_array(np.abs(np.subtract(mean_ex_heatmap, mean_flex_heatmap)))
        channels_flexion,channels_extension = choose_possible_channels(difference_heatmap,mean_flex_heatmap,mean_ex_heatmap)
        return channels_flexion,channels_extension

    def from_emg_grid_to_320_position(self,channels):
        """
        to get from the channels in the grid to the channels in the emg interface list of 320 channels (from 2d to 1d)
        :param channels: List each entrie in lists is [row,col] of channel
        :return: List , each entrie in lists is one number between 0 and 319
        """
        channels = []
        #TODO funktion implementieren
        return channels

    def which_channels_from_emg(self,channels,emng_indices):
        """
        :param channels: List channels_flexion , each entrie in lists is [row,col] of channel
        muss so aufgebaut sein: #movements x 2 (flexion,extension) x #channels(row,col)
        :param emg_indices: list of channels we want to listen to from emg based on grids plugged into emg
        :return: list of channels we want to listen to from emg based on choosen channels + list which listened channel from emg belongs to which movement
        list of channels = im format wie ich es abhören lasse von emg interface
        list channel_movement_assignment = #rows, #cols, überall 0 außer an den stellen wo der channel zu einer bewegung gehört, dort Bewegung und flexion/extension eintragen
        """
        all_important_channels = []
        for movement in channels:
            for movement_phase in movement:
                for channel in movement_phase:
                    all_important_channels.append(channel)

        channels_emg = self.from_emg_grid_to_320_position(all_important_channels)
        #TODO wie channel_movement_assignment aufbauen

        return channels_emg, channel_movement_assignement


import os
import pickle
import scipy.io as sio
import tqdm
import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import torch
from scipy.signal import find_peaks
from scipy.signal import butter
#from ChannelExtraction import ChannelExtraction
import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from exo_controller import ExtractImportantChannels
from scipy.signal import convolve2d
from scipy.signal import resample
from scipy.stats import norm

import ChannelExtraction


def extract_mean_heatmaps(movement_list, path_to_subject_data,emg= None,ref =None):
    """
    if emg and ref will be given, it has to be in format : num_movements x rest
    extracts the mean heatmaps for every movement (including both movement phases flexion and extension)
    :param movement_list:
    :param path_to_subject_data:
    :return:
    """

    mean_heatmaps = {}
    count = 0
    for movement in tqdm.tqdm(movement_list,desc="Extracting mean heatmaps for from all movements"):
        # if emg is None and ref is None:
        #     emg_data,_,ref_data= open_all_files_for_one_patient_and_movement(path_to_subject_data, movement)
        #     extractor = ExtractImportantChannels.ChannelExtraction(movement,emg_data,ref_data)
        # else:
        #     extractor = ExtractImportantChannels.ChannelExtraction(movement, emg[count], ref[count])
        #     count +=1
        extractor = ChannelExtraction.ChannelExtraction(movement, path_to_subject_data)
        heatmap_flexion,heatmap_extension,heatmap_difference = extractor.get_heatmaps()
        mean_heatmaps[movement+ "_flexion"] = heatmap_flexion
        mean_heatmaps[movement+ "_extension"] = heatmap_extension
        mean_heatmaps[movement+ "_difference"] = heatmap_difference
    return mean_heatmaps
def extract_important_channels(movement_list,path_to_subject_dat):
    important_channels = []
    for movement in tqdm.tqdm(movement_list,desc="Extracting important channels for from all movements"):
        extractor = ChannelExtraction.ChannelExtraction(movement, path_to_subject_dat)
        channels = extractor.get_channels()
        for channel in channels:
            if channel not in important_channels:
                important_channels.append(channel)
    return important_channels

def extract_important_channels_realtime(movement_list,emg,ref):
    important_channels = []
    for movement in tqdm.tqdm(movement_list,desc="Extracting important channels for from all movements"):
        extractor = ExtractImportantChannels.ChannelExtraction(movement, emg,ref)
        channels = extractor.get_channels()
        for channel in channels:
            if channel not in important_channels:
                important_channels.append(channel)
    return important_channels
def load_mat_file(file_path, attributes_to_open = None):
    """
    loads a mat file and returns a dict with the attributes as keys or when no keyw are specified, returns the whole dict
    :param file_path:
    :param attributes_to_open:
    :return:
    """
    if attributes_to_open is None:
        mat_data = sio.loadmat(file_path)
        return mat_data
    else:
        mat_data = sio.loadmat(file_path)
        data_dict = {key: mat_data[key] for key in attributes_to_open}
        return data_dict

def save_as_pickle(data,where_to_save):
    """
    saves data as pickle file
    :param data:
    :param where_to_save: this has to be the full path including the filename
    :return:
    """
    with open(where_to_save, 'wb') as f:
        pickle.dump(data, f)

def load_pickle_file(file_path):
    """
    loads a pickle file
    :param file_path:
    :return:
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def open_csv_file(file_path):
    """
    opens a csv file and returns a pandas dataframe
    :param file_path:
    :return:
    """
    data = pandas.read_csv(file_path)
    return data

def plot_spike_train(spike_train, title = None):
    """
    plot spiketrain of MUAPs with input is a 2D array with MUAPs as rows and spiketime as columns
           the spiketime has length of the signal length consisting of 0s and for every sample that has a spike a 1
    :param spike_train:
    :param title:
    :return:
    """
    plt.figure()
    for MU in range(spike_train.shape[0]):
        count = 0
        for spike in spike_train[MU]:
            if (spike != 0):
                plt.vlines(x=count, ymin=MU, ymax=MU + 1)
            count += 1
    plt.title(title)
    plt.show()

def reformat_spiketrain(spike_train):
    """spiketrain for each MU has individual length with every entry is the time of the spike"""
    max_columns = max(arr.shape[1] for arr in spike_train[0, :])
    # Create an empty array to store the stacked arrays
    stacked_array = np.empty((len(spike_train[0, :]), max_columns), dtype=object)
    # Fill the empty array with the arrays from the list
    count = 0
    for i in spike_train[0, :]:
        stacked_array[count, :i.shape[1]] = i[0, :]
        count += 1
    signal_length = spike_train[0][1][0].shape[0]
    result_array = np.zeros((18, signal_length))
    for MU in range(18):
        for pulse in stacked_array[MU]:
            if pulse != None:
                result_array[MU, pulse] = 1
    return result_array

def save_figure(figure, where_to_save):
    """
    saves a figure
    :param figure:
    :param where_to_save:
    :return:
    """
    figure.savefig(where_to_save)

def check_available_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            print("1 GPU is available.")
        else:
            print(f"{num_gpus} GPUs are available.")
    else:
        print("No GPUs are available.")

def open_all_files_for_one_patient_and_movement(folder_path, movement_name):
    """
    opens all files (EMG,MU,ref) for one patient and one movement and returns the data as a dictionary
    :param folder_path: path to the folder that contains data.pkl and the 12 csv files for kinematics
    :param movement_name: which movement you want to open all the files for
    :return: dictionarys for emg_data,MU_data,ref_data
    """
    movement_list = ["thumb_slow", "thumb_fast", "index_slow", "index_fast", "middle_slow", "middle_fast", "ring_slow",
                     "ring_fast", "pinky_slow", "pinky_fast", "fist", "2pinch", "3pinch"]
    movement_number = movement_list.index(movement_name)+1
    file_path = os.path.join(folder_path, "data.pickle")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    emg_data = data[movement_name]["SIG"]
    MU_data = data[movement_name]["MUPulses"]
    ref_file = os.path.join(folder_path,  str(movement_number) + "_transformed.csv")
    ref_data = pandas.read_csv(ref_file)

    return emg_data,MU_data,ref_data

def calculate_emg_rms_one_channel(emg_data):
    """
    Calculate the Root Mean Squared (RMS) of an EMG channel.

    Parameters:
    - emg_data (numpy.ndarray): The EMG data as a 1D NumPy array.

    Returns:
    - float: The RMS value of the EMG data.
    """
    rms = np.sqrt(np.mean(np.square(emg_data)))
    return rms

def calculate_emg_rms_row(emg_grid,position,interval_in_samples):
    """
        Calculate the Root Mean Squared (RMS) for every channel in a 3D grid of EMG channels.

        Parameters:
        - emg_grid (numpy.ndarray): The 3D EMG data grid where the first two dimensions represent rows and columns of channels,
                                   and the third dimension represents the values within each channel.
        - position (int): The position of the EMG grid in the time series.
        -interval_in_samples (int): The number of samples to include in the RMS calculation.

        Returns:
        - numpy.ndarray: An array of RMS values for each channel.
        """

    num_rows = emg_grid.shape[0]
    rms_values = np.zeros((num_rows))

    for row_idx in range(num_rows):
        if (position - interval_in_samples < 0) or (emg_grid[row_idx].shape[0] < (interval_in_samples)):
            channel_data = emg_grid[row_idx][:position + 1]
        else:
            channel_data = emg_grid[row_idx][position - interval_in_samples:position]
        rms_value = np.sqrt(np.mean(np.square(np.array(channel_data))))
        if np.isnan(rms_value):
            print("nan")
            rms_value = 0
        rms_values[row_idx] = rms_value
    return rms_values
def resample_reference_data(ref_data, emg_data):
    """
    resample the reference data such as it has the same length and shape as the emg data
    :param ref_data: dictionary with keys as movement names and values as the reference data
    :param emg_data: dictionary with keys as movement names and values as the emg data
    :return: resampled reference data
    """
    for movement in ref_data.keys():
        ref_data[movement] = resample(ref_data[movement], emg_data[movement].shape[1], axis=0)
    return ref_data
def calculate_emg_rms_grid(emg_grid,position,interval_in_samples):
    """
    Calculate the Root Mean Squared (RMS) for every channel in a 3D grid of EMG channels.

    Parameters:
    - emg_grid (numpy.ndarray): The 3D EMG data grid where the first two dimensions represent rows and columns of channels,
                               and the third dimension represents the values within each channel.
    - position (int): The position of the EMG grid in the time series.
    -interval_in_samples (int): The number of samples to include in the RMS calculation.

    Returns:
    - numpy.ndarray: An array of RMS values for each channel.
    """

    num_rows, num_cols,_ = emg_grid.shape
    rms_values = np.zeros((num_rows, num_cols))

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            if (position-interval_in_samples < 0) or (len(emg_grid[row_idx][col_idx][0]) < (interval_in_samples)):
                channel_data = emg_grid[row_idx][col_idx][0][:position+1]
            else:
                channel_data = emg_grid[row_idx][col_idx][0][position-interval_in_samples:position]
            #print(np.sqrt(np.mean(np.array(channel_data) ** 2)))
            rms_values[row_idx, col_idx] = np.sqrt(np.mean(np.array(channel_data) ** 2))

    return rms_values

def calculate_mu_rms_heatmap_for_one_mu(mu_data):
    """
    Calculate the Root Mean Squared (RMS) of a MU.

    Parameters:
    - mu_data (numpy.ndarray): The MU data as a 3D NumPy array.

    Returns:
    - float: The RMS grid of the MU data.
    """
    num_rows = mu_data.shape[0]
    num_cols = mu_data.shape[1]
    rms_values = np.zeros((num_rows, num_cols))

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            channel_data = mu_data[row_idx][col_idx][:]
            rms_values[row_idx, col_idx] = np.sqrt(np.mean(np.array(channel_data) ** 2))

    return rms_values

def fill_empty_array_indexes_with_0(arr):
    """
    fills empty array indexes with 0 and also returns a mask where the empty indexes are marked as True
    :param arr:
    :return:
    """
    mask = np.zeros_like(arr, dtype=bool)
    for row in range(arr.shape[0]):
        for column in range(arr.shape[1]):
            if len(arr[row,column]) == 0:
                arr[row,column] = [[0]]
                mask[row,column] = True
            else:
                mask[row,column] = False
    return arr,mask

def from_grid_position_to_row_position(grid_position):
    # shape of grid is 13x 26 mit outliers = [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[0,25],[1,25],[2,25],[3,25],[4,25],[5,25],[6,25],[7,25],[12,0],[12,13]]
    outliers = [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[0,25],[1,25],[2,25],[3,25],[4,25],[5,25],[6,25],[7,25],[12,0],[12,13]]

    if grid_position[0] < 8:
        # dann sind es die 8 mal 8 grids 3 nebeneinander
        if grid_position[1] <8:
            grid = 1
            adding = 0
        if grid_position[1] >=8 and grid_position[1] < 16:
            grid = 2
            adding = 8*8*2
        if grid_position[1] >=16:
            grid = 3
            adding = 8*8*3

        row_position = adding + grid_position[0] * 8 + ((grid_position[1]-1) - (grid-1)*8)
    else:
        # dann sind es die 5 mal 13 grids 2 nebeneinander
        # dann sind es die 8 mal 8 grids 3 nebeneinander
        if grid_position[1] < 13:
            grid = 1
            adding = 0
        else:
            grid = 2
            adding = 5*13

        row_position = adding + (grid_position[1]*5 - ((grid - 1) * 13)) + ((grid_position[0] ) )

    if grid_position in outliers:
        print("error in helper function from_grid_position_to_row_position , this positon is an outlier")
    else:
        if row_position > 320:
            print("error in helper function from_grid_position_to_row_position , this positon is out of range")
    return row_position

def find_max_min_values_for_each_movement_and_channel(emg_data,important_channels,movements):
    """
    This function finds the max and min values for each channel in the emg data.
    It will return the max and min values for each channel over all movements.
    This is needed to normalize the data later on.
    :return: max and min values for each channel
    """
    max_values = np.zeros(len(important_channels))
    min_values = np.zeros(len(important_channels))
    for movement in movements:
        for channel in important_channels:
            if np.max(emg_data[movement][channel]) > max_values[channel]:
                max_values[channel] = np.max(emg_data[movement][channel])
            if np.min(emg_data[movement][channel]) < min_values[channel]:
                min_values[channel] = np.min(emg_data[movement][channel])

    return max_values, min_values


def find_q_median_values_for_each_movement_and_channel(emg_data,important_channels,movements):
    """
    This function finds the max and min values for each channel in the emg data.
    It will return the max and min values for each channel over all movements.
    This is needed to normalize the data later on.
    :return: max and min values for each channel
    """
    # TODO wie will ich das hier machen ?? welche median und quantiles nehme ich ??  ich sollte eigentlich genau so machen (größtes 1 und 3 quantile) median keine ahnung :D 
    q1 = np.zeros(len(important_channels))
    q2 = np.zeros(len(important_channels))
    median = np.zeros(len(important_channels))

    for movement in movements:
        for channel in important_channels:
            median[channel] += np.median(emg_data[movement][channel])
            if np.quantile(emg_data[movement][channel],0.25) < q1[channel]:
                q1[channel] = np.quantile(emg_data[movement][channel],0.25)
            if np.quantile(emg_data[movement][channel],0.75) > q2[channel]:
                q2[channel] = np.quantile(emg_data[movement][channel],0.75)

    for i in range(len(important_channels)):
        median[i] = median[i]/len(movements)
    return q1,q2,median

def robust_scaling(data, q1,q2,median):
    """
    scales the data with robust scaling
    :param data:
    :param q1:
    :param q2:
    :param median:
    :return:
    """
    return (np.array(data) - np.array(median)) / ((np.array(q2) - np.array(q1)))
def normalize_2D_array(data,axis=None,negative = False, max_value = None, min_value = None):
    """
    normalizes a 2D array

    :param data: the data to be normalized
    :param axis: which axis should be normalized, if None, the whole array will be normalized if 0 columns will be normalized, if 1 rows will be normalized
    :param negative: if True, the data will be normalized between -1 and 1, if False, the data will be normalized between 0 and 1
    :param max_value: if given we do not want to make max/ min scaling on this sample but for every channel based on max min of the whole trainings data
    :param min_value: if given we do not want to make max/ min scaling on this sample but for every channel based on max min of the whole trainings data
    :return:
    """
    data = np.array(data)
    if (max_value is not None) and (min_value is not None):

        min_value = np.array(min_value)
        max_value = np.array(max_value)
        norm = (data - min_value) / ((max_value - min_value) )

    elif axis is None:
        data = np.array(data)
        norm = (data-np.min(data))/(np.max(data)-np.min(data))
    else:
        data = np.array(data)
        norm = (data-np.min(data,axis=axis))/(np.max(data,axis=axis)-np.min(data,axis=axis))

    if negative == True:
        norm = (norm * 2) - 1
    return norm



def spikeTriggeredAveraging(SIG, MUPulses, STA_window, emg_length, fsamp):
    """
    Calculate the spike triggered average (STA) for a given spike train and EMG signal.
    :param SIG:
    :param MUPulses:
    :param STA_window:
    :param emg_length:
    :param fsamp:
    :return:
    """

    # Convert STA_window to samples
    STA_window = round((STA_window / 2) * fsamp)
    STA_mean = np.zeros((MUPulses.shape[1], SIG.shape[0], SIG.shape[1],2*STA_window))  # Initialize STA_mean with #of MUs, #rows, and #columns
    for MU in range(MUPulses.shape[1]):  # For each MU in the data...
        MU_STA = []  # Initialize MU_STA for this MU
        for row in range(SIG.shape[0]):  # ... extract for each row ...
            row_STA = []  # Initialize row_STA for this row
            for col in range(SIG.shape[1]):  # ... and column of the grid ...
                if len(SIG[row][col][0]) >1:  # Check if the channel is not empty
                    temp_STA = []  # Initialize temp_STA
                    for spks in MUPulses[0,MU][0]:  # ... the raw EMG signal of that channel inside the STA window around each MU firing.
                        if (spks + STA_window < emg_length) and (spks - STA_window >= 0): # check if the STA window is within the emg signal
                            temp_STA.append(SIG[row][col][0][spks - STA_window:spks + STA_window]) # fill temp STA with the emg signal around the MU firing for all spikes

                    STA_mean[MU][row][col][:] = np.nanmean(np.array(temp_STA).reshape(len(temp_STA),len(temp_STA[0])),axis=0)  # Compute mean of all these extracted EMG signals
    return STA_mean

def check_to_which_movement_cycle_sample_belongs(sample_position,local_maxima,local_minima,method = "closest_extrema"):
    """
    Checks to which movement cycle a sample belongs to (i.e. the sample is closer to the next local maxima or minima)
    chooses the maxima/minima like marius mentioned (if sample is after minima it belongs to the next maxima)
    :param sample_position:
    :param local_maxima:
    :param local_minima:
    :return:
    """

    # Find the nearest local maxima and minima to the sample position
    closest_maxima = min(local_maxima, key=lambda x: abs(sample_position - x))
    closest_minima = min(local_minima, key=lambda x: abs(sample_position - x))

    if method == "marius":
        # 1 = flexion , 2 = extension
        if sample_position < closest_maxima:
            return 1 , abs(sample_position - closest_maxima)
        else:
            return 2 , abs(sample_position - closest_minima)
    elif method == "closest_extrema":
        if abs(sample_position - closest_maxima) < abs(sample_position - closest_minima):
            return 1 , abs(sample_position - closest_maxima)
        else:
            return 2 , abs(sample_position - closest_minima)
    else:
        print("wrong method given")

def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def plot_predictions(ground_truth,prediction,tree_number= None,realtime = False):
    """
    Plots ground truth and predictions from two regression models for two movement regressions.

    Args:
        ground_truth (numpy.ndarray): The ground truth values for both regressions.
        prediction (numpy.ndarray): The predicted values for both regressions.
        tree_number (int): The number of the tree that was used to make the predictions.
    """
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)
    plt.figure(figsize=(12, 6))
    num_rows= ground_truth.shape[0]
    x_values= list(range(num_rows))
    if realtime == True:
        prediction = prediction.squeeze(axis=1)

    if ground_truth.ndim == 1:
        for i in range(ground_truth.shape[0]):
            plt.plot([i,i],[ground_truth[i],prediction[i]],color='grey')
        plt.scatter(x_values,ground_truth, label='Ground Truth', color='blue',s=10)
        plt.scatter(x_values,prediction, label='Prediction', color='green', s=10)

        plt.legend()
        plt.title('Ground Truth vs. Predictions for tree number : ' + str(tree_number))
        plt.xlabel('Sample Index')
        plt.ylabel('Values')
        plt.tight_layout()
        plt.ylim(0,1)
        plt.show()

    else:
        plt.subplot(1, 2, 1)
        for i in range(ground_truth.shape[0]):
            plt.plot([i,i],[ground_truth[i,0],prediction[i,0]],color='grey')
        plt.scatter(x_values,ground_truth[:,0], label='Ground Truth flexion', color='blue',s=10)
        plt.scatter(x_values,prediction[:,0], label='flexion value prediction', color='green', s=10)

        plt.legend()
        plt.title('Ground Truth 1 vs. Predictions for tree number : ' + str(tree_number))
        plt.xlabel('Sample Index')
        plt.ylim(0, 1)
        plt.ylabel('Values')

        plt.subplot(1, 2, 2)
        for i in range(ground_truth.shape[0]):
            plt.plot([i,i],[ground_truth[i,1],prediction[i,1]],color='grey')
        plt.scatter(x_values,ground_truth[:,1], label='Ground Truth extension', color='blue',s=10)
        plt.scatter(x_values,prediction[:,1], label='extension value prediction', color='green',s=10)

        plt.legend()
        plt.title('Ground Truth 2 vs. Predictions for tree number : ' + str(tree_number))
        plt.xlabel('Sample Index')
        plt.ylabel('Values')

        # Displaying the plot
        plt.tight_layout()
        plt.ylim(0, 1)
        plt.show()

def get_locations_of_all_maxima(movement_signal, distance=2800):
    """
    returns the locations of all local maxima and minima in the movement signal, distance is the minimum distance between two maxima/minima
    :param movement_signal:
    :param distance:
    :return:
    """

    if movement_signal.ndim == 2:
        if movement_signal.shape[1] == 1:
            movement_signal = movement_signal.squeeze(axis=1)
        else:
            movement_signal = movement_signal[:,0]

    local_maxima,_ = find_peaks(movement_signal,distance=distance)
    local_minima,_ = find_peaks(-movement_signal,distance=distance)

    return local_maxima, local_minima

def plot_local_maxima_minima(movement_signal,local_maxima,local_minima,current_sample_position=None,color = 'black'):
    """
    plots the local maxima and minima on top of the movement signal (maxima with red crosses, minima with green crosses)
    :param movement_signal:
    :param local_maxima:
    :param local_minima:
    :param current_sample_position: if given, plots a cross in color: color at the current sample position
    :return:
    """
    plt.figure()
    plt.plot(movement_signal)
    plt.plot(local_maxima,movement_signal[local_maxima],'x',color='red')
    plt.plot(local_minima,movement_signal[local_minima],'x',color='green')
    if current_sample_position is not None:
        plt.plot(current_sample_position,movement_signal[current_sample_position],'x',color=color)
    plt.show()


def choose_possible_channels(difference_heatmap,mean_flex_heatmap,mean_ex_heatmap,threshold_neighbours=0.25, threshold_difference_amplitude=0.35):
    """
    Chooses the channels that are more active in one movement than in the other
    difference_heatmap: heatmap of the difference between the two movements
    mean_flex_heatmap: heatmap of the mean of the flexion movement
    mean_ex_heatmap: heatmap of the mean of the extension movement
    threshold: only areas where at least one neighboring channel the slope to current value is below this threshold will be considered (low threshold = small slope = equal intesity at neighboring channels)
    :param difference_heatmap:
    :param mean_flex_heatmap:
    :param mean_ex_heatmap:
    :param threshold:
    :return:
    """
    #threshold = only areas with values above it will be considered


    ex_list = []
    flex_list = []
    #TODO jede listen eintrag besteht aus [row,col]
    #extract important channels from different heatmap
    #then search for important channels in both means , for the one where activity is higher than in the other fill channel to list
    # Determine whether the channel is more active in movement 1 or movement 2
    # Find the centroids of each area and calculate the channel with the highest activity
    #TODO additionaly filter for outliers (has high activity but surrounding not)

    # Iterate through the grid
    for i in range(difference_heatmap.shape[0]):
        for j in range(difference_heatmap.shape[1]):
            # Initialize a flag to check if at least one neighbor has a lower slope
            neighbours= []
            differences = []
            for x_offset in [-1,0, 1]:
                for y_offset in [-1,0, 1]:
                    if 0 <= i + x_offset < difference_heatmap.shape[0] and 0 <= j + y_offset < difference_heatmap.shape[1]:
                        if(x_offset==0 and y_offset==0):
                            continue
                        neighbours.append(difference_heatmap[i + x_offset, j + y_offset])
                        differences.append(np.abs(difference_heatmap[i + x_offset, j + y_offset] - difference_heatmap[i, j]))
            max_neighbour = max(neighbours)
            this_value = difference_heatmap[i,j]

            if (min(differences)<= threshold_neighbours) and (this_value > max_neighbour) and ( this_value > threshold_difference_amplitude):
                activity_in_movement_1 = mean_flex_heatmap[i][j]
                activity_in_movement_2 = mean_ex_heatmap[i][j]

                # Assign the channel to the respective list based on activity
                if activity_in_movement_1 > activity_in_movement_2:
                    flex_list.append([i, j])
                else:
                    ex_list.append([i, j])

            if (this_value<= min(neighbours) and (min(differences)<= threshold_neighbours) and (this_value is not None) and (this_value !=  0.0)):
                activity_in_movement_1 = mean_flex_heatmap[i][j]
                activity_in_movement_2 = mean_ex_heatmap[i][j]
                #print("refference channel found for low acitivity")
                # Assign the channel to the respective list based on activity
                if activity_in_movement_1 < activity_in_movement_2:
                    flex_list.append([i, j])
                else:
                    ex_list.append([i, j])


    return flex_list, ex_list

def reshape_emg_channels_into_320(emg_data):
    """
    reshapes the emg channels into 320 channels x number of samples
    :param emg_data:
    :return:
    """
    reshaped_emg_data = []
    for row in range (emg_data.shape[0]):
        for col in range (emg_data.shape[1]):
            if emg_data[row,col].shape[0] > 0:
                reshaped_emg_data.append(emg_data[row,col] [0])

    reshaped_emg_data = np.array(reshaped_emg_data).reshape(320,-1)
    return reshaped_emg_data

def is_near_extremum(frame,local_maxima, local_minima,time_window,sampling_frequency):
    """
    Checks if the frame is near a local maxima or minima (i.e. the frame is within a certain time window of the local maxima or minima)
    :param frame:
    :param local_maxima:
    :param local_minima:
    :return:
    """
    time_window_in_samples = (time_window/1000) * sampling_frequency
    # Find the nearest local maxima and minima to the frame
    closest_maxima = min(local_maxima, key=lambda x: abs(frame - x))
    closest_minima = min(local_minima, key=lambda x: abs(frame - x))
    # Calculate the distances from the frame to the closest maxima and minima
    distance_to_maxima = abs(frame - closest_maxima)
    distance_to_minima = abs(frame - closest_minima)

    # Check if the frame is near a local maxima or minima
    if distance_to_maxima < time_window_in_samples or distance_to_minima < time_window_in_samples:
        return True
    else:
        return False

def plot_emg_channels(emg_data, shift=1400,save_path=None,movement=None):
    """
    Plots each EMG channel with a vertical shift.

    :param emg_data: A 2D numpy array with shape (channels, timepoints)
    :param shift: The vertical shift to apply between each channel
    :param save_path: The path to save the figure
    :param movement: The movement name
    :return: None
    """
    emg_data= emg_data[:,0:10*2048]
    n_channels, n_timepoints = emg_data.shape
    time = np.arange(n_timepoints) / 2048

    fig = plt.figure(figsize=(10, 25))
    fig.patch.set_visible(False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.ylim(-300, n_channels * shift + shift)
    # Plot each channel
    for i in range(n_channels):
        # Apply a vertical shift to each channel
        shifted_data = emg_data[i, :] + i * shift
        plt.plot(time, shifted_data, label=f'Channel {i+1}',linewidth=0.25)

    plt.xlabel('Time in s')
    plt.ylabel('EMG signal + Shift')
    plt.grid()
    plt.title('EMG Signals Over Time')
    plt.tight_layout()
    fig_name = os.path.join(save_path,  str(movement) + ".pdf")
    plt.savefig(fig_name, transparent=True)
    plt.close()
    #plt.show()


def create_gaussian_filter(size_filter=3,sigma=None):
    """
    Creates a Gaussian filter of the given size and standard deviation.

    :param size: The size of the filter (size x size)
    :param sigma: The standard deviation of the Gaussian filter. If None, sigma is set to size/6.
    :return: A 2D numpy array representing the Gaussian filter
    """
    if sigma is None:
        sigma = size_filter / 6.0  # A rule of thumb is to set sigma to 1/6 of the kernel size

    kernel = np.ones((size_filter, size_filter))
    for row in range(size_filter):
        for col in range(size_filter):
            x = col - size_filter // 2
            y = row - size_filter // 2
            kernel[row, col] = np.multiply(np.divide(1, 2 * np.pi* sigma**2),
                                           np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))

    # Normalize the kernel to ensure the sum of all elements is 1
    kernel /= np.sum(kernel)

    return kernel


def split_grid_into_8x8_grids(grid):
    """
    Splits the grid into 8x8 grids.

    :param grid: A numpy array representing the grid i.e with shape f.e.(16, 27)
    :return: A list of 8x8 numpy arrays
    """
    # Initialize an empty list to store the 8x8 grids
    grids = []

    # Iterate over each row in the grid
    for i in range(0, grid.shape[0], 8):
        # Iterate over each column in the grid
        for j in range(0, grid.shape[1], 8):
            # Extract the 8x8 grid
            grids.append(grid[i:i + 8, j:j + 8])

    return grids


def apply_gaussian_filter(grid, gaussian_filter):
    """
    Applies the Gaussian filter to the grid to smooth the values.

    :param grid: A 16x27 numpy array representing the grid
    :param gaussian_filter: A 2D numpy array representing the Gaussian filter
    :return: A 16x27 numpy array with smoothed values
    """

    # Define the function to handle the border
    grid = grid.reshape(grid.shape[0],grid.shape[1])
    grids = split_grid_into_8x8_grids(grid.reshape(grid.shape[0],grid.shape[1]))
    def filter_function(image, filter):
        filtered_image = np.zeros_like(image)
        filter_size = filter.shape[0]
        pad_width = filter_size // 2

        # Pad the image with the edge values to handle borders
        padded_image = np.pad(image, pad_width, mode='edge')

        # Iterate over each cell in the original image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Define the region of interest in the padded image
                region = padded_image[i:i + filter_size, j:j + filter_size]
                # Apply the filter, ignoring the parts that go beyond the borders
                filtered_value = np.sum(region * filter)
                filtered_image[i, j] = filtered_value

        return filtered_image

    for one_grid in range(len(grids)):
        if one_grid <3: # if upper grids
            grid[0:8,0+8*one_grid:8+8*one_grid] = filter_function(grids[one_grid], gaussian_filter)
        else: # if lower grids
            grid[8:16, 0 + 8 * (one_grid-3):8 + 8 * (one_grid-3)] = filter_function(grids[one_grid], gaussian_filter)

    return grid





#area to test functions
if __name__ == "__main__":
    # Define the parameters
    num_periods = 5  # Number of periods
    num_samples_per_period = 100  # Number of samples per period
    frequency = 1.0  # Frequency of the sinusoidal signal (in Hz)

    # Define a list of amplitudes for each period
    amplitudes = [1.0, 0.8, 0.6, 0.4, 0.2]

    # Calculate the total number of samples
    num_samples = num_periods * num_samples_per_period

    # Generate the time values
    t = np.linspace(0, num_periods / frequency, num_samples)

    # Initialize an empty array to store the signal
    signal = np.zeros(num_samples)

    # Generate the signal with varying amplitudes
    for i in range(num_periods):
        start_idx = i * num_samples_per_period
        end_idx = (i + 1) * num_samples_per_period
        signal[start_idx:end_idx] = amplitudes[i] * np.sin(2 * np.pi * frequency * t[start_idx:end_idx])


    sample_position = 49  # Replace with the desired sample position

    maxima,minima = get_locations_of_all_maxima(signal)
    result = check_to_which_movement_cycle_sample_belongs(sample_position,maxima,minima)
    plt.figure()
    plt.scatter(maxima,signal[maxima],c='r')
    plt.scatter(minima,signal[minima],c='b')
    plt.plot(signal)
    plt.scatter(sample_position,signal[sample_position],c='g')
    plt.scatter(result,signal[result],c='y')

    plt.show()

    # print(f"The sample is closer to the {result}.")
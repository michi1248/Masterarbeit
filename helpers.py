import os
import pickle
import scipy.io as sio
import tqdm
import pandas
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import convolve2d


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
    """opens all files (EMG,MU,ref) for one patient and one movement and returns the data as a dict"""
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
    rms = np.sqrt(np.mean(emg_data**2))
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

        rms_values[row_idx] = np.sqrt(np.mean(np.array(channel_data) ** 2))
    return rms_values

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

    num_rows, num_cols = emg_grid.shape
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



def normalize_2D_array(data):
    """
    normalizes a 2D array
    :param array:
    :return:
    """
    data = np.array(data)
    norm = (data-np.min(data))/(np.max(data)-np.min(data))
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

def check_to_which_movement_cycle_sample_belongs(sample_position,local_maxima,local_minima):
    """
    Checks to which movement cycle a sample belongs to (i.e. the sample is closer to the next local maxima or minima)
    :param sample_position:
    :param local_maxima:
    :param local_minima:
    :return:
    """
    # Find the nearest local maxima and minima to the sample position
    closest_maxima = min(local_maxima, key=lambda x: abs(sample_position - x))
    closest_minima = min(local_minima, key=lambda x: abs(sample_position - x))
    # Calculate the distances from the sample to the closest maxima and minima
    distance_to_maxima = abs(sample_position - closest_maxima)
    distance_to_minima = abs(sample_position - closest_minima)

    # 1 = flexion , 2 = extension
    if distance_to_maxima < distance_to_minima:
        return 1 , distance_to_maxima
    else:
        return 2 , distance_to_minima

def get_locations_of_all_maxima(movement_signal):
    # Calculate the local maxima and minima of the signal
    local_maxima = np.where((movement_signal[:-2] < movement_signal[1:-1]) & (movement_signal[1:-1] > movement_signal[2:]))[0] + 1
    local_minima = np.where((movement_signal[:-2] > movement_signal[1:-1]) & (movement_signal[1:-1] < movement_signal[2:]))[0] + 1
    return local_maxima,local_minima

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
            else:
                print("Slope difference too high or other value in neighborhood is higher")
                print("row: ", i, " col: ", j)



        # # Limit the number of areas to the specified value
        # if len(important_channels_movement_1) >= num_areas and len(important_channels_movement_2) >= num_areas:
        #     break

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
    Checks if the frame is near a local maxima or minima
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
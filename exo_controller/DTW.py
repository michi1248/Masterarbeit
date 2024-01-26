import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy import signal
from scipy.signal import butter, filtfilt
from dtw import *


def align_signals_dtw(rms_emg_signal, movement_signal,emg_grid):

    """Align the EMG signal with the movement signal using DTW.
    Parameters
    ----------
    emg_signal : rms of all emg channels (rms over each channel for each time step)
    movement_signal : the movement signal (shape [#samples,2])
    """

    if rms_emg_signal is None or movement_signal is None:
        raise ValueError("Both EMG and movement signals must be set before alignment.")

    if np.max(movement_signal[:,0]) >  np.max(movement_signal[:, 1]) :
        print("used thumb (0)")
        signal_to_compare = movement_signal[:,0]
        index = 0
    else:
        print("used index (1)")
        signal_to_compare = movement_signal[:,1]
        index = 1


    original_emg_length = len(rms_emg_signal)
    original_movement_length = len(signal_to_compare)
    downsampled_emg = resample(rms_emg_signal, len(rms_emg_signal)//10)
    downsampled_movement = resample(movement_signal, len(signal_to_compare)//10, axis=0)
    emg_grid = resample(emg_grid, len(rms_emg_signal)//10, axis=1)

    print("computing dtw")
    alignment = dtw(downsampled_emg,downsampled_movement[:,index], keep_internals=True,step_pattern=rabinerJuangStepPattern(6, "c"),open_begin=False,open_end=False)
    print("dtw done")

    aligned_emg =  np.array(downsampled_emg[alignment.index1])
    aligned_movement = np.array(downsampled_movement[alignment.index2,:] )
    emg_grid = np.array(emg_grid[:,alignment.index1])

    aligned_emg = resample(aligned_emg, original_emg_length)
    aligned_movement = resample(aligned_movement, original_movement_length, axis=0)
    emg_grid = resample(emg_grid, original_emg_length, axis=1)

    # Extracting the aligned signals based on the DTW path
    # aligned_emg = np.array(rms_emg_signal[index_mapping[0]])
    # aligned_movement = np.array(movement_signal[index_mapping[1], :])
    # emg_grid = emg_grid[:, index_mapping[0]]

    plt.figure()
    plt.subplot(211)
    plt.plot(rms_emg_signal,label="original emg")
    plt.plot(aligned_emg, label="aligned emg")
    plt.legend()

    plt.subplot(212)
    plt.plot(signal_to_compare,label="original movement")
    plt.plot(aligned_movement[:,index],label="aligned movement")
    plt.legend()
    plt.show()
    return emg_grid, aligned_movement

def butter_lowpass_filter(signal, cutoff_freq= 20, fs=2048, order = 2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def make_rms_for_dtw(emg_matrix):
    """Compute the RMS of the EMG matrix.
    -> (rms over each channel for each time step)
    Parameters
    ----------
    emg_matrix : the EMG matrix (shape [#channels, #samples])
    """
    for channel in range(emg_matrix.shape[0]):
        emg_matrix[channel] = butter_lowpass_filter(emg_matrix[channel])

    rms = np.sqrt(np.median(emg_matrix**2, axis=0))
    #rms = butter_lowpass_filter(rms,cutoff_freq=0.2)


    return rms
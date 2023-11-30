import os

import numpy as np

from exo_controller.helpers import *
from exo_controller import grid_arrangement
from scipy.signal import fftconvolve

import seaborn as sns
from PIL import Image
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from exo_controller.helpers import *
from scipy.signal import resample
from exo_controller import *


class Heatmap:
    def __init__(
        self,
        movement_name,
        path_to_subject_dat,
        sampling_frequency=2048,
        path_to_save_plots=r"D:\Lab\differences_train_test_heatmaps",
        frame_duration=150,
        additional_term="",
        method="roboust",
        mean_flex_rest=None,
        mean_ex_rest=None,
        spatial_filtering=False,
        bandpass_filter=False,
    ):
        """
        NOTE !!!! at the moment if there are more than one ref dimension is given (2pinch,fist,etc,) for the plot the first one is taken!!!
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
        print(
            "NOTE !!!! at the moment if there are more than one ref dimension is given (2pinch,fist,etc,) for the plot the first one is taken!!!",
            file=sys.stderr,
        )
        self._DIFFERENTIAL_FILTERS = {
            "identity": np.array([[1]]),  # identity case when no filtering is applied
            "LSD": np.array([[-1], [1]]),  # longitudinal single differential
            "LDD": np.array([[1], [-2], [1]]),  # longitudinal double differential
            "TSD": np.array([[-1, 1]]),  # transverse single differential
            "TDD": np.array([[1, -2, 1]]),  # transverse double differential
            "NDD": np.array(
                [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
            ),  # normal double differential or Laplacian filter
            "IB2": np.array(
                [[-1, -2, -1], [-2, 12, -2], [-1, -2, -1]]
            ),  # inverse binomial filter of order 2
            "IR": np.array(
                [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
            ),  # inverse rectangle
        }
        if method == "Gauss_filter":
            self.gauss_filter = create_gaussian_filter(size_filter=5)
        self.movement_name = movement_name
        self.mean_ex = mean_ex_rest
        self.mean_flex = mean_flex_rest
        if bandpass_filter == True:
            additional_term = additional_term + "_bandpass"
        if spatial_filtering == True:
            additional_term = additional_term + "_spatial"
        if not os.path.exists(
            os.path.join(
                path_to_save_plots,
                str(frame_duration) + "ms_rms_window",
                method,
                path_to_subject_dat.split("/")[1] + "_" + additional_term,
            )
        ):
            os.makedirs(
                os.path.join(
                    path_to_save_plots,
                    str(frame_duration) + "ms_rms_window",
                    method,
                    path_to_subject_dat.split("/")[1] + "_" + additional_term,
                )
            )

        grid_aranger = grid_arrangement.Grid_Arrangement([2, 3, 4, 1, 5])
        grid_aranger.make_grid()
        self.path_to_save_plots = os.path.join(
            path_to_save_plots,
            str(frame_duration) + "ms_rms_window",
            method,
            path_to_subject_dat.split("/")[1] + "_" + additional_term,
        )
        self.emg_data = (
            load_pickle_file(os.path.join(path_to_subject_dat, "emg_data.pkl"))[
                movement_name
            ]
            .transpose(1, 0, 2)
            .reshape(320, -1)
        )
        if spatial_filtering == True:
            grid = grid_aranger.transfer_320_into_grid_arangement(self.emg_data)
            grid = grid_aranger.concatenate_upper_and_lower_grid(grid[0], grid[1])
            filtered_grid = self.spatial_filtering(grid)
            self.emg_data = grid_aranger.transfer_grid_arangement_into_320(
                filtered_grid
            )
        if bandpass_filter == True:
            before = self.emg_data
            after = bandpass_filter_emg_data(self.emg_data, fs=2048)
            self.emg_data = after

        # emg_data_for_max_min = load_pickle_file(os.path.join(path_to_subject_dat, "emg_data.pkl"))
        all_emg_data = []

        for movement in [
            "subject_rest",
            "subject_2pinch_low_extension",
            "subject_2pinch_low_flexion",
            "subject_2pinch_strong_extension",
            "subject_2pinch_strong_flexion",
            "subject_index_low_extension",
            "subject_index_low_flexion",
            "subject_Index_strong_extension",
            "subject_Index_strong_flexion",
            "subject_thumb_strong_extension",
            "subject_thumb_strong_flexion",
        ]:
            path = (
                r"D:\Lab\MasterArbeit\trainings_data\resulting_trainings_data\test_if_works/"
                + str(movement)
            )  # trainingsdata recorded after training
            a = (
                load_pickle_file(os.path.join(path, "emg_data.pkl"))["rest"]
                .transpose(1, 0, 2)
                .reshape(320, -1)[64:256]
            )
            if (self.mean_ex is not None) and (self.mean_flex is not None):
                # have to transfer self.mean_ex from grid arrangement to 320 channels arangement
                a = grid_aranger.transfer_320_into_grid_arangement1(a)[0] - np.reshape(
                    self.mean_ex, (self.mean_ex.shape[0], self.mean_ex.shape[1], 1)
                )
            else:
                a = grid_aranger.transfer_320_into_grid_arangement1(a)[0]
            all_emg_data.append(a)
        self.all_emg_data = all_emg_data

        # for i in emg_data_for_max_min.keys():
        #     emg_data_for_max_min[i] = np.array(emg_data_for_max_min[i].transpose(1,0,2).reshape(320,-1))[64:256]
        #
        #     if (self.mean_ex is not None) and (self.mean_flex is not None):
        #         # have to transfer self.mean_ex from grid arrangement to 320 channels arangement
        #
        #         emg_data_for_max_min[i] = emg_data_for_max_min[i] - grid_aranger.transfer_grid_arangement_into_320(np.reshape(self.mean_ex,(self.mean_ex.shape[0],self.mean_ex.shape[1],1)))
        #
        #     if (method == "Bandpass_filter") or (bandpass_filter == True):
        #         before = emg_data_for_max_min[i]
        #         after = bandpass_filter_emg_data(emg_data_for_max_min[i], fs=2048)
        #         emg_data_for_max_min[i] = after
        #
        #     if (method == "Spatial_EMG_signals") or (spatial_filtering == True):
        #         grid = grid_aranger.transfer_320_into_grid_arangement(emg_data_for_max_min[i])
        #         grid = grid_aranger.concatenate_upper_and_lower_grid(grid[0],grid[1])
        #         filtered_grid = self.spatial_filtering(grid)
        #         emg_data_for_max_min[i] = grid_aranger.transfer_grid_arangement_into_320(filtered_grid)
        #
        # if method == "RMS":
        #
        #     first_grid = np.sqrt(np.mean(np.square(self.emg_data[64:128]), axis=0))
        #     second_grid = np.sqrt(np.mean(np.square(self.emg_data[128:192]), axis=0))
        #     third_grid = np.sqrt(np.mean(np.square(self.emg_data[192:256]), axis=0))
        #     fig,ax = plt.subplots(3,1, figsize=(15,10), sharey=True,sharex=True)
        #
        #     x = np.linspace(0, first_grid.shape[0] / 2048, first_grid.shape[0])
        #     ax[0].plot(x,first_grid, label="first grid")
        #     ax[0].set_title("Grid 1")
        #
        #
        #     ax[1].plot(x,second_grid, label="second grid")
        #     ax[1].set_title("Grid 2")
        #
        #     ax[2].plot(x,third_grid, label="third grid")
        #     ax[2].set_title("Grid 3")
        #
        #     fig.suptitle("RMS of all 3 grids")
        #     fig.supxlabel("Time (s)")#
        #     fig.supylabel("Amplitude (mV)")
        #     plt.title("Grid 3")
        #     plt.savefig(os.path.join(self.path_to_save_plots, "rms.png"))
        #
        #
        # if method == "EMG_signals" :
        #     self.max_values, self.min_values = find_max_min_values_for_each_movement_and_channel(all_emg_data,
        #                                                                                          range(320), list(
        #             emg_data_for_max_min.keys()))
        #     self.max_values = self.max_values.reshape(320, 1)
        #     self.min_values = self.min_values.reshape(320, 1)
        #     self.q1, self.q2, self.median = find_q_median_values_for_each_movement_and_channel(all_emg_data,
        #                                                                                        range(320), list(
        #             emg_data_for_max_min.keys()))
        #     self.q1 = self.q1.reshape(320, 1)
        #     self.q2 = self.q2.reshape(320, 1)
        #     self.median = self.median.reshape(320, 1)
        #
        #
        #     plot_emg_channels(emg_data_for_max_min[self.movement_name][64:256], save_path=self.path_to_save_plots,
        #                       movement=self.movement_name + "_raw")
        #
        #     plot_emg_channels(robust_scaling(emg_data_for_max_min[self.movement_name], self.q1, self.q2, self.median)[64:256],
        #                       save_path=self.path_to_save_plots, movement=self.movement_name + "_robust",shift=3)
        #     plot_emg_channels(
        #         normalize_2D_array(emg_data_for_max_min[self.movement_name], max_value=self.max_values, min_value=self.min_values)[64:256],
        #         save_path=self.path_to_save_plots, movement=self.movement_name + "_min_max",shift=1)
        #
        # if  method == "Spatial_EMG_signals":
        #     self.max_values, self.min_values = find_max_min_values_for_each_movement_and_channel(all_emg_data,
        #                                                                                          range(320), list(
        #             emg_data_for_max_min.keys()))
        #     self.max_values = self.max_values.reshape(320, 1)
        #     self.min_values = self.min_values.reshape(320, 1)
        #     self.q1, self.q2, self.median = find_q_median_values_for_each_movement_and_channel(all_emg_data,
        #                                                                                        range(320), list(
        #             emg_data_for_max_min.keys()))
        #     self.q1 = self.q1.reshape(320, 1)
        #     self.q2 = self.q2.reshape(320, 1)
        #     self.median = self.median.reshape(320, 1)
        #
        #     plot_emg_channels(emg_data_for_max_min[self.movement_name][64:256], save_path=self.path_to_save_plots,
        #                       movement=self.movement_name + "_raw")
        #
        #     plot_emg_channels(
        #         robust_scaling(emg_data_for_max_min[self.movement_name], self.q1, self.q2, self.median)[64:256],
        #         save_path=self.path_to_save_plots, movement=self.movement_name + "_robust", shift=3)
        #     plot_emg_channels(
        #         normalize_2D_array(emg_data_for_max_min[self.movement_name], max_value=self.max_values,
        #                            min_value=self.min_values)[64:256],
        #         save_path=self.path_to_save_plots, movement=self.movement_name + "_min_max", shift=1)
        #
        # if method == "Bandpass_filter":
        #     self.max_values, self.min_values = find_max_min_values_for_each_movement_and_channel(all_emg_data,
        #                                                                                          range(320), list(
        #             emg_data_for_max_min.keys()))
        #     self.max_values = self.max_values.reshape(320, 1)
        #     self.min_values = self.min_values.reshape(320, 1)
        #     self.q1, self.q2, self.median = find_q_median_values_for_each_movement_and_channel(all_emg_data,
        #                                                                                        range(320), list(
        #             emg_data_for_max_min.keys()))
        #     self.q1 = self.q1.reshape(320, 1)
        #     self.q2 = self.q2.reshape(320, 1)
        #     self.median = self.median.reshape(320, 1)
        #
        #     plot_emg_channels(emg_data_for_max_min[self.movement_name][64:256], save_path=self.path_to_save_plots,
        #                       movement=self.movement_name + "_raw")
        #
        #     plot_emg_channels(
        #         robust_scaling(emg_data_for_max_min[self.movement_name], self.q1, self.q2, self.median)[64:256],
        #         save_path=self.path_to_save_plots, movement=self.movement_name + "_robust", shift=3)
        #     plot_emg_channels(
        #         normalize_2D_array(emg_data_for_max_min[self.movement_name], max_value=self.max_values,
        #                            min_value=self.min_values)[64:256],
        #         save_path=self.path_to_save_plots, movement=self.movement_name + "_min_max", shift=1)

        upper, lower = grid_aranger.transfer_320_into_grid_arangement(self.emg_data)

        self.ref_data = load_pickle_file(
            os.path.join(path_to_subject_dat, "3d_data.pkl")
        )[movement_name]
        resampling_factor = 2048 / 120
        num_samples_emg = self.emg_data.shape[1]
        num_samples_resampled_ref = int(self.ref_data.shape[0] * resampling_factor)
        resampled_ref = np.empty((num_samples_resampled_ref, 2))

        self.ref_data = resample(self.ref_data, num_samples_resampled_ref)

        self.emg_data = (
            upper  # grid_aranger.concatenate_upper_and_lower_grid(upper,lower)
        )

        self.sampling_frequency = sampling_frequency

        self.sample_length = self.emg_data[1][1].shape[0]
        self.num_samples = int(
            self.sample_length / (self.sampling_frequency * (frame_duration / 1000))
        )
        self.frame_duration = frame_duration
        self.number_observation_samples = int(
            (self.frame_duration / 1000) * self.sampling_frequency
        )
        # self.ax_ref.set_title("Reference")
        # self.ax_ref.set(xlabel="Time (s)", ylabel="Amplitude (mV)",xlim=(0, self.sample_length / self.sampling_frequency), ylim=(0, 1))
        self.fig, (self.ax_1, self.ax_2, self.ax_3) = plt.subplots(
            3, 1, figsize=(8, 15)
        )
        self.fig.suptitle("Heatmmap comparison of " + movement_name, fontsize=14)
        self.x_for_ref = np.linspace(
            0, self.sample_length / self.sampling_frequency, self.ref_data.shape[0]
        )
        self.global_counter = 0
        self.last_frame = 0

    def spatial_filtering(self, siganl_to_be_filtered, filter_name="NDD"):
        """This function applies the filters to the chunk."""

        # Extend filter dimensions prior to performing a convolution
        # flt_coeff = np.expand_dims(self._DIFFERENTIAL_FILTERS[filter_name], axis=(0, 1, -1))
        grid = siganl_to_be_filtered
        grids = split_grid_into_8x8_grids(grid)
        for one_grid in range(len(grids)):
            if one_grid < 3:  # if upper grids
                grid[0:8, 0 + 8 * one_grid : 8 + 8 * one_grid, :] = fftconvolve(
                    grids[one_grid],
                    self._DIFFERENTIAL_FILTERS[filter_name].reshape(
                        (
                            self._DIFFERENTIAL_FILTERS[filter_name].shape[0],
                            self._DIFFERENTIAL_FILTERS[filter_name].shape[1],
                            1,
                        )
                    ),
                    mode="same",
                ).astype(np.float32)
            else:  # if lower grids
                grid[
                    8:16, 0 + 8 * (one_grid - 3) : 8 + 8 * (one_grid - 3), :
                ] = fftconvolve(
                    grids[one_grid],
                    self._DIFFERENTIAL_FILTERS[filter_name].reshape(
                        (
                            self._DIFFERENTIAL_FILTERS[filter_name].shape[0],
                            self._DIFFERENTIAL_FILTERS[filter_name].shape[1],
                            1,
                        )
                    ),
                    mode="same",
                ).astype(
                    np.float32
                )

        return grid

    def get_max_min_values(self, emg_data, channels):
        """
        :return: max and min values of the emg data
        """
        max_values = []
        min_values = []
        for channel in channels:
            max_values.append(np.max(emg_data[channel]))
            min_values.append(np.min(emg_data[channel]))
        return max_values, min_values

    def get_median_quantile_values(self, emg_data, channels):
        q1_values = []
        q2_values = []
        median_values = []
        for channel in channels:
            q1_values.append(np.quantile(emg_data[channel], 0.25))
            q2_values.append(np.quantile(emg_data[channel], 0.75))
            median_values.append(np.median(emg_data[channel]))
        return q1_values, q2_values, median_values

    def calculate_heatmap(self, emg_grid, position, interval_in_samples):
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

        num_rows, num_cols, _ = emg_grid.shape
        rms_values = np.zeros((num_rows, num_cols))

        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                if (position - interval_in_samples < 0) or (
                    len(emg_grid[row_idx][col_idx]) < (interval_in_samples)
                ):
                    channel_data = emg_grid[row_idx][col_idx][: position + 1]
                else:
                    channel_data = emg_grid[row_idx][col_idx][
                        position - interval_in_samples : position
                    ]
                # print(np.sqrt(np.mean(np.array(channel_data) ** 2)))
                rms_values[row_idx, col_idx] = np.sqrt(
                    np.mean(np.array(channel_data) ** 2)
                )
        return rms_values

    def make_heatmap_emg(self, frame):
        """
        :return: heatmap of emg data

        """

        heatmap = self.calculate_heatmap(
            self.emg_data, frame, self.number_observation_samples
        )

        if (self.mean_ex is not None) and (self.mean_flex is not None):
            heatmap = heatmap - self.mean_ex
        if method == "Gauss_filter":
            normalized_heatmap = robust_scaling(
                heatmap,
                q1=self.q1[:, :, 0],
                q2=self.q2[:, :, 0],
                median=self.median[:, :, 0],
            )
            normalized_heatmap = apply_gaussian_filter(
                normalized_heatmap, self.gauss_filter
            )

        if method == "no_scaling":
            normalized_heatmap = heatmap

        if method == "Min_Max_Scaling_over_whole_data":
            normalized_heatmap = normalize_2D_array(
                heatmap,
                max_value=self.max_values[:, :, 0],
                min_value=self.min_values[:, :, 0],
            )

        if method == "Robust_Scaling":
            normalized_heatmap = robust_scaling(
                heatmap,
                q1=self.q1[:, :, 0],
                q2=self.q2[:, :, 0],
                median=self.median[:, :, 0],
            )

        if method == "Min_Max_Scaling_over_sample":
            normalized_heatmap = normalize_2D_array(heatmap)

        if method == "Min_Max_Scaling_all_channels":
            normalized_heatmap = normalize_2D_array(
                heatmap, max_value=self.max_values, min_value=self.min_values
            )

        if method == "Robust_all_channels":
            normalized_heatmap = robust_scaling(
                heatmap, q1=self.q1, q2=self.q2, median=self.median
            )

        if self.movement_name == "rest":
            self.heatmaps_flex = np.add(self.heatmaps_flex, heatmap)
            # add heatmap and not normalized heatmap because impact of all heatmaps will be the same but maybe some heatmaps have higher values and i want to use this ??
            # TODO maybe change this (better to use normalized or not ?)
            # IT IS BETER TO USE NORMALIZED BECAUSE IF EMG SCHWANKUNGEN
            self.number_heatmaps_flex += 1
            self.heatmaps_ex = np.add(self.heatmaps_ex, heatmap)
            self.number_heatmaps_ex += 1

        # only do the following if +- window size near extrema
        if (self.movement_name != "rest") and (
            is_near_extremum(
                frame,
                self.local_maxima,
                self.local_minima,
                time_window=1.5 * self.frame_duration,
                sampling_frequency=self.sampling_frequency,
            )
        ):
            # find out to which part of the movement the current sample belongs (half of flex or ref)
            (
                belongs_to_movement,
                distance,
            ) = check_to_which_movement_cycle_sample_belongs(
                frame, self.local_maxima, self.local_minima
            )

            # add the heatmap to the list of all heatmaps of the fitting flex/ex for later calculation of difference heatmap
            if belongs_to_movement == 1:
                # the closer the sample is to the extrema the more impact it has on the heatmap
                self.heatmaps_flex = np.add(
                    self.heatmaps_flex, heatmap
                )  # np.multiply(heatmap, 1/(distance+0.1) ))
                # add heatmap and not normalized heatmap because impact of all heatmaps will be the same but maybe some heatmaps have higher values and i want to use this ??
                # TODO maybe change this (better to use normalized or not ?)
                # IT IS BETER TO USE NORMALIZED BECAUSE IF EMG SCHWANKUNGEN
                self.number_heatmaps_flex += 1
                plot_local_maxima_minima(
                    self.ref_data[:, 0],
                    self.local_maxima,
                    self.local_minima,
                    current_sample_position=frame,
                    color="black",
                )
            else:
                self.heatmaps_ex = np.add(
                    self.heatmaps_ex, heatmap
                )  # np.multiply(heatmap, 1/(distance+0.1) ))
                self.number_heatmaps_ex += 1
                plot_local_maxima_minima(
                    self.ref_data[:, 0],
                    self.local_maxima,
                    self.local_minima,
                    current_sample_position=frame,
                    color="yellow",
                )

        if self.global_counter == 0:
            hmap = sns.heatmap(
                normalized_heatmap[:, :8],
                ax=self.ax_1,
                cmap="RdBu_r",
                cbar=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={"label": "RMS"},
            )  # ,cbar_ax=self.ax_emg,)
            hmap = sns.heatmap(
                normalized_heatmap[:, 8:16],
                ax=self.ax_2,
                cmap="RdBu_r",
                cbar=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={"label": "RMS"},
            )  # ,cbar_ax=self.ax_emg,)
            hmap = sns.heatmap(
                normalized_heatmap[:, 16:],
                ax=self.ax_3,
                cmap="RdBu_r",
                cbar=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={"label": " RMS"},
            )  # ,cbar_ax=self.ax_emg,)
        else:
            hmap = sns.heatmap(
                normalized_heatmap[:, :8],
                ax=self.ax_1,
                cmap="RdBu_r",
                cbar=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={"label": "RMS"},
            )  # ,cbar_ax=self.ax_emg,)
            hmap = sns.heatmap(
                normalized_heatmap[:, 8:16],
                ax=self.ax_2,
                cmap="RdBu_r",
                cbar=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={"label": " RMS"},
            )  # ,cbar_ax=self.ax_emg,)
            hmap = sns.heatmap(
                normalized_heatmap[:, 16:],
                ax=self.ax_3,
                cmap="RdBu_r",
                cbar=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={"label": " RMS"},
            )  # ,cbar_ax=self.ax_emg,)
        self.global_counter += 1
        # self.emg_plot.set(normalized_heatmap, ax=self.ax_emg, cmap='hot', cbar=True, xticklabels=True, yticklabels=True, cbar_kws={'label': 'norm. RMS'},mask=mask_for_heatmap)
        # self.ax_emg.imshow(heatmap, cmap='hot', interpolation='nearest')

    def make_Ref_trackings(self, frame):
        if (
            ("pinch" in self.movement_name)
            or ("fist" in self.movement_name)
            or ("rest" in self.movement_name)
        ):
            self.ax_ref.plot(
                self.x_for_ref, normalize_2D_array(self.ref_data[:, 0]), color="blue"
            )
            self.ax_ref.scatter(
                self.x_for_ref[frame],
                normalize_2D_array(self.ref_data[:, 0])[frame],
                color="green",
                marker="x",
                s=90,
                linewidth=3,
            )
        else:
            self.ax_ref.plot(
                self.x_for_ref, normalize_2D_array(self.ref_data), color="blue"
            )
            self.ax_ref.scatter(
                self.x_for_ref[frame],
                normalize_2D_array(self.ref_data)[frame],
                color="green",
                marker="x",
                s=90,
                linewidth=3,
            )

    def update(self, frame):
        plt.figure()
        self.fig, (self.ax_1, self.ax_2, self.ax_3) = plt.subplots(
            3, 1, figsize=(8, 15)
        )
        self.fig.suptitle(
            "Heatmmap comparison of "
            + self.movement_name
            + "at sample: "
            + str(frame)
            + " with window size "
            + str(self.frame_duration)
            + " ms",
            fontsize=12,
        )

        self.make_heatmap_emg(frame)

        # Save the frame as an image
        frame_filename = os.path.join(self.path_to_save_plots, f"frame_{frame:03d}.png")
        self.fig.canvas.draw()
        # plt.savefig(frame_filename, bbox_inches='tight', dpi=100)
        pil_image = Image.frombytes(
            "RGB", self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb()
        )
        pil_image.save(frame_filename)
        self.pbar.update(1)

        return self.ax_1, self.ax_2, self.ax_3

    def update_slider(self, event):
        print(self.slider.get())
        self.update(int(self.slider.get()))

    def close_plot(self):
        self.root.quit()
        self.root.destroy()

    def move_to_closest(self):
        current_value = int(self.slider.get())
        print("current value is: " + str(current_value))
        closest_value = min(self.samples, key=lambda x: abs(x - current_value))
        index_of_closest_value = self.samples.index(closest_value)
        next_higher_value = self.samples[index_of_closest_value + 1]
        print("next value is: " + str(next_higher_value))
        self.slider.set(next_higher_value)

    def animate(self, save=False):
        print("number_observation_samples: " + str(self.number_observation_samples))
        # samples = all sample values when using all samples with self.frame_duration in between
        self.samples = np.linspace(
            self.number_observation_samples,
            self.sample_length,
            self.num_samples,
            endpoint=False,
            dtype=int,
        )
        self.samples = [
            element for element in self.samples if element <= self.ref_data.shape[0]
        ]
        # make both lists to save all coming heatmaps into it by adding the values and dividing at the end through number of heatmaps
        num_rows, num_cols, _ = self.emg_data.shape
        self.heatmaps_flex = np.zeros((num_rows, num_cols))
        self.heatmaps_ex = np.zeros((num_rows, num_cols))
        self.number_heatmaps_flex = 0
        self.number_heatmaps_ex = 0
        if (
            ("pinch" in self.movement_name)
            or ("fist" in self.movement_name)
            or ("rest" in self.movement_name)
        ):
            self.local_maxima, self.local_minima = get_locations_of_all_maxima(
                self.ref_data[:, 0], distance=5000
            )
            plot_local_maxima_minima(
                self.ref_data[:, 0], self.local_maxima, self.local_minima
            )
        else:
            self.local_maxima, self.local_minima = get_locations_of_all_maxima(
                self.ref_data[:]
            )
            plot_local_maxima_minima(
                self.ref_data, self.local_maxima, self.local_minima
            )

        if not save:
            # stuff for interactive plot
            self.root = tk.Tk()
            self.root.title("Interactive Heatmap Plot")
            self.label = ttk.Label(self.root, text="Sample")
            self.label.pack()
            self.slider = tk.Scale(
                self.root,
                from_=0,
                to=self.sample_length,
                orient=tk.HORIZONTAL,
                length=600,
                command=self.update_slider,
            )
            self.slider.set(self.samples[0])
            self.slider.pack()
            # Create a "Next" button to trigger the move_to_closest function
            next_button = tk.Button(
                self.root, text="Next", command=self.move_to_closest
            )
            next_button.pack()
            # self.slider = widgets.SelectionSlider(options=samples, value=0, description='Sample', continuous_update=False)
            # self.interactive_plot = widgets.interact(self.update, frame=slider)
            # self.slider.bind("<ButtonRelease-1>", self.update_slider)
            # self.slider.bind("<ScaleChanged>", self.update_slider)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
            self.canvas.get_tk_widget().pack()
            # Create a button to close the plot window
            close_button = ttk.Button(
                self.root, text="Close Plot", command=self.close_plot
            )
            close_button.pack()
            # Start the Tkinter main loop
            self.root.mainloop()

        if save:
            # for tqdm
            num_frames = len(self.samples)
            self.pbar = tqdm.tqdm(total=num_frames, position=0, leave=True)
            self.ani = FuncAnimation(
                self.fig, self.update, frames=self.samples[:], cache_frame_data=False
            )
            # self.ani.event_source.stop()
        plt.show()

    def calculate_heatmaps_all(self):
        if method == "Min_Max_Scaling_all_channels":
            self.max_values = -10000000
            self.min_values = 10000000

        if method == "Robust_Scaling":
            self.q1 = (
                np.zeros((self.emg_data.shape[0], self.emg_data.shape[1])) - 10000000
            )
            self.q2 = (
                np.zeros((self.emg_data.shape[0], self.emg_data.shape[1])) + 10000000
            )
            self.median = np.zeros((self.emg_data.shape[0], self.emg_data.shape[1]))

        if method == "Robust_all_channels":
            self.q1 = -1000000000
            self.q2 = 10000000000
            self.median = 0

        if method == "Min_Max_Scaling_over_whole_data":
            self.max_values = (
                np.zeros((self.emg_data.shape[0], self.emg_data.shape[1])) - 10000000
            )
            self.min_values = (
                np.zeros((self.emg_data.shape[0], self.emg_data.shape[1])) + 10000000
            )

        all_heatmaps = []

        for movement in range(len(self.all_emg_data)):
            self.samples = np.linspace(
                self.number_observation_samples,
                self.all_emg_data[movement][1][1].shape[0],
                self.num_samples,
                endpoint=False,
                dtype=int,
            )

            for frame in tqdm.tqdm(self.samples):
                heatmap = self.calculate_heatmap(
                    self.all_emg_data[movement], frame, self.number_observation_samples
                )
                if (self.mean_ex is not None) and (self.mean_flex is not None):
                    heatmap = heatmap - self.mean_ex

                if method == "Min_Max_Scaling_all_channels":
                    if np.max(heatmap) > self.max_values:
                        self.max_values = np.max(heatmap)
                    if np.min(heatmap) < self.min_values:
                        self.min_values = np.min(heatmap)

                if method == "Min_Max_Scaling_over_whole_data":
                    for row in range(heatmap.shape[0]):
                        for col in range(heatmap.shape[1]):
                            if heatmap[row, col] > self.max_values[row, col]:
                                self.max_values[row, col] = heatmap[row, col]
                            if heatmap[row, col] < self.min_values[row, col]:
                                self.min_values[row, col] = heatmap[row, col]

                if "Robust" in method:
                    all_heatmaps.append(heatmap)

        if method == "Min_Max_Scaling_over_whole_data":
            self.max_values = np.reshape(
                self.max_values, (self.max_values.shape[0], self.max_values.shape[1], 1)
            )
            self.min_values = np.reshape(
                self.min_values, (self.min_values.shape[0], self.min_values.shape[1], 1)
            )

        if method == "Robust_Scaling":
            all_heatmaps = np.stack(np.array(all_heatmaps), axis=-1)
            self.q1 = np.quantile(all_heatmaps, 0.25, axis=-1)
            self.q2 = np.quantile(all_heatmaps, 0.75, axis=-1)
            self.median = np.median(all_heatmaps, axis=-1)

            self.q1 = np.reshape(self.q1, (self.q1.shape[0], self.q1.shape[1], 1))
            self.q2 = np.reshape(self.q2, (self.q2.shape[0], self.q2.shape[1], 1))
            self.median = np.reshape(
                self.median, (self.median.shape[0], self.median.shape[1], 1)
            )

        if method == "Robust_all_channels":
            all_heatmaps = np.stack(np.array(all_heatmaps), axis=-1)
            self.q1 = np.min(np.quantile(all_heatmaps, 0.25, axis=-1))
            self.q2 = np.max(np.quantile(all_heatmaps, 0.75, axis=-1))
            self.median = np.mean(np.median(all_heatmaps, axis=-1))

    def channel_extraction(self, mark_choosen_channels=False):
        self.pbar.close()
        ########
        if method == "Gauss_filter":
            mean_flex_heatmap = robust_scaling(
                np.divide(self.heatmaps_flex, self.number_heatmaps_flex),
                q1=self.q1[:, :, 0],
                q2=self.q2[:, :, 0],
                median=self.median[:, :, 0],
            )
            mean_ex_heatmap = robust_scaling(
                np.divide(self.heatmaps_ex, self.number_heatmaps_ex),
                q1=self.q1[:, :, 0],
                q2=self.q2[:, :, 0],
                median=self.median[:, :, 0],
            )

            mean_flex_heatmap = apply_gaussian_filter(
                mean_flex_heatmap, self.gauss_filter
            )
            mean_ex_heatmap = apply_gaussian_filter(mean_ex_heatmap, self.gauss_filter)

        if method == "Min_Max_Scaling_over_whole_data":
            mean_flex_heatmap = normalize_2D_array(
                np.divide(self.heatmaps_flex, self.number_heatmaps_flex),
                max_value=self.max_values[:, :, 0],
                min_value=self.min_values[:, :, 0],
            )
            mean_ex_heatmap = normalize_2D_array(
                np.divide(self.heatmaps_ex, self.number_heatmaps_ex),
                max_value=self.max_values[:, :, 0],
                min_value=self.min_values[:, :, 0],
            )
        if method == "Robust_Scaling":
            # mean_flex_heatmap = np.divide(self.heatmaps_flex, self.number_heatmaps_flex)
            # mean_ex_heatmap = np.divide(self.heatmaps_ex, self.number_heatmaps_ex)
            mean_flex_heatmap = robust_scaling(
                np.divide(self.heatmaps_flex, self.number_heatmaps_flex),
                q1=self.q1[:, :, 0],
                q2=self.q2[:, :, 0],
                median=self.median[:, :, 0],
            )
            mean_ex_heatmap = robust_scaling(
                np.divide(self.heatmaps_ex, self.number_heatmaps_ex),
                q1=self.q1[:, :, 0],
                q2=self.q2[:, :, 0],
                median=self.median[:, :, 0],
            )
        if method == "Min_Max_Scaling_over_sample":
            mean_flex_heatmap = normalize_2D_array(
                np.divide(self.heatmaps_flex, self.number_heatmaps_flex)
            )
            mean_ex_heatmap = normalize_2D_array(
                np.divide(self.heatmaps_ex, self.number_heatmaps_ex)
            )

        if method == "no_scaling":
            mean_flex_heatmap = np.divide(self.heatmaps_flex, self.number_heatmaps_flex)
            mean_ex_heatmap = np.divide(self.heatmaps_ex, self.number_heatmaps_ex)

        if method == "Min_Max_Scaling_all_channels":
            mean_flex_heatmap = normalize_2D_array(
                np.divide(self.heatmaps_flex, self.number_heatmaps_flex),
                max_value=self.max_values,
                min_value=self.min_values,
            )
            mean_ex_heatmap = normalize_2D_array(
                np.divide(self.heatmaps_ex, self.number_heatmaps_ex),
                max_value=self.max_values,
                min_value=self.min_values,
            )
        if method == "Robust_all_channels":
            mean_flex_heatmap = robust_scaling(
                np.divide(self.heatmaps_flex, self.number_heatmaps_flex),
                q1=self.q1,
                q2=self.q2,
                median=self.median,
            )
            mean_ex_heatmap = robust_scaling(
                np.divide(self.heatmaps_ex, self.number_heatmaps_ex),
                q1=self.q1,
                q2=self.q2,
                median=self.median,
            )

        mean_ex_heatmap[np.isnan(mean_ex_heatmap)] = 0
        mean_flex_heatmap[np.isnan(mean_flex_heatmap)] = 0
        # mean_ex_heatmap = normalize_2D_array(mean_ex_heatmap) # here normalize again just over this sample because it is nevertheless just  one image
        # mean_flex_heatmap = normalize_2D_array(mean_flex_heatmap)
        # difference_heatmap = normalize_2D_array(np.abs(np.subtract(mean_ex_heatmap, mean_flex_heatmap)))
        difference_heatmap = np.subtract(mean_ex_heatmap, mean_flex_heatmap)
        difference_heatmap[np.isnan(difference_heatmap)] = 0
        # difference_heatmap = normalize_2D_array(difference_heatmap)
        if mark_choosen_channels:
            channels_flexion, channels_extension = choose_possible_channels(
                difference_heatmap, mean_flex_heatmap, mean_ex_heatmap
            )

            print("choosen channels flexion: ", channels_flexion)
            print("choosen channels extension: ", channels_extension)
            chosen_heatmap_flexion = np.zeros(
                (mean_flex_heatmap.shape[0], mean_flex_heatmap.shape[1])
            )
            for i in channels_flexion:
                chosen_heatmap_flexion[i[0]][i[1]] = 1
            chosen_heatmap_extension = np.zeros(
                (mean_ex_heatmap.shape[0], mean_ex_heatmap.shape[1])
            )
            for i in channels_extension:
                chosen_heatmap_extension[i[0]][i[1]] = 1
        path_to_difference_plot = os.path.join(
            self.path_to_save_plots, "difference_plots"
        )
        if not os.path.exists(path_to_difference_plot):
            os.makedirs(path_to_difference_plot)

        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        # difference heatmap

        global_min = np.min(mean_flex_heatmap)
        global_max = np.max(mean_flex_heatmap)

        axes[0].set_title("Grid 1 ")
        sns.heatmap(
            mean_flex_heatmap[:, :8],
            ax=axes[0],
            cmap="RdBu_r",
            vmin=global_min,
            vmax=global_max,
            cbar=True,
        )
        axes[1].set_title("Grid 2 ")
        sns.heatmap(
            mean_flex_heatmap[:, 8:16],
            ax=axes[1],
            cmap="RdBu_r",
            vmin=global_min,
            vmax=global_max,
            cbar=True,
        )
        axes[2].set_title("Grid 3 ")
        sns.heatmap(
            mean_flex_heatmap[:, 16:],
            ax=axes[2],
            cmap="RdBu_r",
            vmin=global_min,
            vmax=global_max,
            cbar=True,
        )

        plt.savefig(
            os.path.join(
                path_to_difference_plot, "difference_plot_" + str(movement) + ".png"
            )
        )
        return np.divide(self.heatmaps_flex, self.number_heatmaps_flex), np.divide(
            self.heatmaps_ex, self.number_heatmaps_ex
        )
        # chosen channels flexion heatmap
        # sns.heatmap(chosen_heatmap_flexion,ax=axes[1])
        # plt.subplot(3, 1)
        # # chosen channels flexion heatmap
        # sns.heatmap(chosen_heatmap_extension, ax=axes[2])
        # plt.savefig(os.path.join(path_to_difference_plot, "difference_plot" + str(movement) + ".png"))
        ####

    def save_animation(self, output_filename, fps=10):
        self.ani.save(output_filename, writer="pillow", fps=fps)


if __name__ == "__main__":
    # a = grid_arrangement.Grid_Arrangement([1,2,3,4,5])
    # a.make_grid()
    # con = a.concatenate_upper_and_lower_grid(np.reshape(a.upper_grids,(a.upper_grids.shape[0],a.upper_grids.shape[1],1)),np.reshape(a.lower_grids,(a.lower_grids.shape[0],a.lower_grids.shape[1],1)))
    # gauss_kernel = create_gaussian_filter(sigma=0.45)
    # filtered = apply_gaussian_filter(con,gauss_kernel)
    plot_emg = False
    plot_spatial_emg = False
    plot_bandpass_filter = False
    rms = False

    # "Min_Max_Scaling_over_whole_data" = min max scaling with max/min is choosen individually for every channel
    # "Robust_all_channels" = robust scaling with q1,q2,median is choosen over all channels
    # "Robust_Scaling"  = robust scaling with q1,q2,median is choosen individually for every channel
    # "Min_Max_Scaling_all_channels" = min max scaling with max/min is choosen over all channels
    # "EMG_signals" = plot emg signals
    # "Gauss_filter" = use Gauss filter on raw data
    # "Bandpass_filter" = use bandpass filter on raw data and plot the filtered all emg channels

    window_size = 150
    for method in [
        "Min_Max_Scaling_all_channels",
        "Robust_all_channels",
        "Robust_Scaling",
        "Min_Max_Scaling_over_whole_data",
    ]:
        mean_flex_rest = None
        mean_ex_rest = None
        # for movement in ['subject_rest', 'subject_thumb_strong_extension', 'subject_thumb_strong_flexion']:
        for movement in [
            "subject_rest",
            "subject_2pinch_low_extension",
            "subject_2pinch_low_flexion",
            "subject_2pinch_strong_extension",
            "subject_2pinch_strong_flexion",
            "subject_index_low_extension",
            "subject_index_low_flexion",
            "subject_Index_strong_extension",
            "subject_Index_strong_flexion",
            "subject_thumb_strong_extension",
            "subject_thumb_strong_flexion",
        ]:
            path = (
                r"D:\Lab\MasterArbeit\trainings_data\resulting_trainings_data\test_if_works/"
                + str(movement)
            )  # trainingsdata recorded after training

            print("method is: ", method)
            print("movement is: ", movement)

            heatmap = Heatmap(
                "rest",
                path,
                frame_duration=window_size,
                method=method,
                bandpass_filter=False,
            )
            heatmap.calculate_heatmaps_all()
            heatmap.animate(save=True)
            fps = 2
            heatmap.save_animation(
                r"D:\Lab\differences_train_test_heatmaps/"
                + str(window_size)
                + "ms_rms_window/"
                + method
                + "/"
                + str(movement)
                + str(fps)
                + "fps_most_firings.gif",
                fps=2,
            )
            mean_flex, mean_ex = heatmap.channel_extraction()
            heatmap.channel_extraction(mark_choosen_channels=True)

            if plot_emg:
                # Do the same again for method == EMG_signals to produce emg singals plot
                heatmap = Heatmap(
                    "rest", path, frame_duration=window_size, method="EMG_signals"
                )
            if plot_spatial_emg:
                heatmap = Heatmap(
                    "rest",
                    path,
                    frame_duration=window_size,
                    method="Spatial_EMG_signals",
                )
            if plot_bandpass_filter:
                heatmap = Heatmap(
                    "rest", path, frame_duration=window_size, method="Bandpass_filter"
                )
            if rms:
                heatmap = Heatmap(
                    "rest", path, frame_duration=window_size, method="RMS"
                )

            if "rest" in movement:
                # if the movement is rest save the mean flex and ex for later subtraction from the other movements
                mean_ex_rest = mean_ex
                mean_flex_rest = mean_flex

            else:
                # if the movement is not rest subtract the mean flex and ex from the rest from the current movement therefore give the means of the rest to the functions
                heatmap = Heatmap(
                    "rest",
                    path,
                    frame_duration=window_size,
                    additional_term="_subtracted_mean_rest_from_emg",
                    method=method,
                    mean_flex_rest=mean_flex_rest,
                    mean_ex_rest=mean_ex_rest,
                    bandpass_filter=False,
                )
                heatmap.calculate_heatmaps_all()
                heatmap.animate(save=True)
                heatmap.save_animation(
                    r"D:\Lab\differences_train_test_heatmaps/"
                    + str(window_size)
                    + "ms_rms_window/"
                    + method
                    + "/"
                    + str(movement)
                    + str(fps)
                    + "fps_most_firings.gif",
                    fps=2,
                )
                mean_flex, mean_ex = heatmap.channel_extraction()
                heatmap.channel_extraction(mark_choosen_channels=True)

                if plot_emg:
                    # Do the same again for method == EMG_signals to produce emg singals plot
                    heatmap = Heatmap(
                        "rest",
                        path,
                        frame_duration=window_size,
                        additional_term="_subtracted_mean_rest_from_emg",
                        method="EMG_signals",
                        mean_flex_rest=mean_flex_rest,
                        mean_ex_rest=mean_ex_rest,
                    )
                if plot_spatial_emg:
                    heatmap = Heatmap(
                        "rest",
                        path,
                        frame_duration=window_size,
                        additional_term="_subtracted_mean_rest_from_emg",
                        method="Spatial_EMG_signals",
                        mean_flex_rest=mean_flex_rest,
                        mean_ex_rest=mean_ex_rest,
                    )
                if plot_bandpass_filter:
                    heatmap = Heatmap(
                        "rest",
                        path,
                        frame_duration=window_size,
                        additional_term="_subtracted_mean_rest_from_emg",
                        method="Bandpass_filter",
                        mean_flex_rest=mean_flex_rest,
                        mean_ex_rest=mean_ex_rest,
                    )
                if rms:
                    heatmap = Heatmap(
                        "rest",
                        path,
                        frame_duration=window_size,
                        additional_term="_subtracted_mean_rest_from_emg",
                        method="RMS",
                        mean_flex_rest=mean_flex_rest,
                        mean_ex_rest=mean_ex_rest,
                    )

    # movement = "thumb"
    # heatmap = Heatmap(movement,r"D:\Lab\MasterArbeit\trainings_data\resulting_trainings_data\subject_Michi_Test1",additional_term="before")
    # heatmap.animate(save=True)
    #
    # fps = 2
    # heatmap.save_animation(r"D:\Lab\differences_train_test_heatmaps/" +str(movement) +str(fps) +"fps_most_firings.gif", fps=2)
    # heatmap.channel_extraction()
    # heatmap.channel_extraction(mark_choosen_channels=True)

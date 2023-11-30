import numpy as np
import tqdm
from exo_controller import helpers


class ChannelExtraction:
    def __init__(
        self,
        movement_name,
        path_to_subject_dat,
        sampling_frequency=2048,
        frame_duration=150,
    ):
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
        self.important_channels = []
        (
            self.emg_data,
            self.Mu_data,
            self.ref_data,
        ) = helpers.open_all_files_for_one_patient_and_movement(
            path_to_subject_dat, movement_name
        )
        self.emg_data, self.mask_for_heatmap = helpers.fill_empty_array_indexes_with_0(
            self.emg_data
        )
        self.ref_data = self.ref_data.to_numpy()
        self.sample_length = self.emg_data[0][1][0].shape[0]
        self.sampling_frequency = sampling_frequency
        self.num_samples = int(
            self.sample_length / (self.sampling_frequency * (frame_duration / 1000))
        )
        self.frame_duration = frame_duration
        self.number_observation_samples = int(
            (self.frame_duration / 1000) * self.sampling_frequency
        )
        self.global_counter = 0
        self.last_frame = 0
        self.closest_mu = 0

    def make_heatmap_emg(self, frame):
        """
        :return: heatmap of emg data

        """
        heatmap = helpers.calculate_emg_rms_grid(
            self.emg_data, frame, self.number_observation_samples
        )
        normalized_heatmap = helpers.normalize_2D_array(heatmap)

        # only do the following if +- window size near extrema
        if helpers.is_near_extremum(
            frame,
            self.local_maxima,
            self.local_minima,
            time_window=self.frame_duration,
            sampling_frequency=self.sampling_frequency,
        ):
            # find out to which part of the movement the current sample belongs (half of flex or ref)
            (
                belongs_to_movement,
                distance,
            ) = helpers.check_to_which_movement_cycle_sample_belongs(
                frame, self.local_maxima, self.local_minima
            )
            # add the heatmap to the list of all heatmaps of the fitting flex/ex for later calculation of difference heatmap
            if belongs_to_movement == 1:
                # the closer the sample is to the extrema the more impact it has on the heatmap
                self.heatmaps_flex = np.add(
                    self.heatmaps_flex,
                    np.multiply(normalized_heatmap, 1 / (distance + 0.1)),
                )
                # add heatmap and not normalized heatmap because impact of all heatmaps will be the same but maybe some heatmaps have higher values and i want to use this ??
                # TODO maybe change this (better to use normalized or not ?)
                # IT IS BETER TO USE NORMALIZED BECAUSE IF EMG SCHWANKUNGEN
                self.number_heatmaps_flex += 1
            else:
                self.heatmaps_ex = np.add(
                    self.heatmaps_ex,
                    np.multiply(normalized_heatmap, 1 / (distance + 0.1)),
                )
                self.number_heatmaps_ex += 1

        self.global_counter += 1

    def heatmap_extraction(self):
        if self.number_heatmaps_flex > 0:
            mean_flex_heatmap = helpers.normalize_2D_array(
                np.divide(self.heatmaps_flex, self.number_heatmaps_flex)
            )
        if self.number_heatmaps_ex > 0:
            mean_ex_heatmap = helpers.normalize_2D_array(
                np.divide(self.heatmaps_ex, self.number_heatmaps_ex)
            )
        if self.number_heatmaps_flex > 0 and self.number_heatmaps_ex > 0:
            difference_heatmap = helpers.normalize_2D_array(
                np.abs(np.subtract(mean_ex_heatmap, mean_flex_heatmap))
            )
        return mean_flex_heatmap, mean_ex_heatmap, difference_heatmap

    def get_heatmaps(self):
        # samples = all sample values when using all samples with self.frame_duration in between
        self.samples = np.linspace(
            0, self.sample_length, self.num_samples, endpoint=False, dtype=int
        )
        self.samples = [
            element for element in self.samples if element <= self.ref_data.shape[0]
        ]
        # make both lists to save all coming heatmaps into it by adding the values and dividing at the end through number of heatmaps
        num_rows, num_cols = self.emg_data.shape
        self.heatmaps_flex = np.zeros((num_rows, num_cols))
        self.heatmaps_ex = np.zeros((num_rows, num_cols))
        self.number_heatmaps_flex = 0
        self.number_heatmaps_ex = 0
        self.local_maxima, self.local_minima = helpers.get_locations_of_all_maxima(
            self.ref_data[:]
        )
        important_channels = []
        heatmap_list = {}
        for i in self.samples:
            self.make_heatmap_emg(i)

        heatmap_flex, heatmap_ex, heatmap_difference = self.heatmap_extraction()
        return heatmap_flex, heatmap_ex, heatmap_difference

    def move_to_closest(self):
        current_value = int(self.slider.get())
        print("current value is: " + str(current_value))
        closest_value = min(self.samples, key=lambda x: abs(x - current_value))
        index_of_closest_value = self.samples.index(closest_value)
        next_higher_value = self.samples[index_of_closest_value + 1]
        print("next value is: " + str(next_higher_value))

    def get_channels(self):
        # samples = all sample values when using all samples with self.frame_duration in between
        self.samples = np.linspace(
            0, self.sample_length, self.num_samples, endpoint=False, dtype=int
        )
        self.samples = [
            element for element in self.samples if element <= self.ref_data.shape[0]
        ]
        # make both lists to save all coming heatmaps into it by adding the values and dividing at the end through number of heatmaps
        num_rows, num_cols = self.emg_data.shape
        self.heatmaps_flex = np.zeros((num_rows, num_cols))
        self.heatmaps_ex = np.zeros((num_rows, num_cols))
        self.number_heatmaps_flex = 0
        self.number_heatmaps_ex = 0
        self.local_maxima, self.local_minima = helpers.get_locations_of_all_maxima(
            self.ref_data[:]
        )
        important_channels = []
        for i in self.samples:
            self.make_heatmap_emg(i)
            channels_flexion, channels_extension = self.channel_extraction()
            for j in channels_flexion:
                if j not in important_channels:
                    important_channels.append(j)

            for j in channels_extension:
                if j not in important_channels:
                    important_channels.append(j)

        return important_channels

    def channel_extraction(self):
        channels_flexion, channels_extension = [], []
        if self.number_heatmaps_flex > 0:
            mean_flex_heatmap = helpers.normalize_2D_array(
                np.divide(self.heatmaps_flex, self.number_heatmaps_flex)
            )
        if self.number_heatmaps_ex > 0:
            mean_ex_heatmap = helpers.normalize_2D_array(
                np.divide(self.heatmaps_ex, self.number_heatmaps_ex)
            )
        if self.number_heatmaps_flex > 0 and self.number_heatmaps_ex > 0:
            difference_heatmap = helpers.normalize_2D_array(
                np.abs(np.subtract(mean_ex_heatmap, mean_flex_heatmap))
            )
            channels_flexion, channels_extension = helpers.choose_possible_channels(
                difference_heatmap, mean_flex_heatmap, mean_ex_heatmap
            )

        return channels_flexion, channels_extension


if __name__ == "__main__":
    movement_list = [
        "thumb_slow",
        "thumb_fast",
        "index_slow",
        "index_fast",
    ]
    important_channels = []
    for movement in tqdm.tqdm(movement_list):
        extractor = ChannelExtraction(movement, r"D:\Lab\data\extracted\Sub2")
        channels = extractor.get_channels()
        for channel in channels:
            if channel not in important_channels:
                important_channels.append(channel)

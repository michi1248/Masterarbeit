import numpy as np
import tqdm
from exo_controller import helpers
from exo_controller import normalizations


class ChannelExtraction:
    def __init__(
        self,
        movement_name,
        emg,
        ref,
        sampling_frequency=2048,
        frame_duration=150,
        use_gaussian_filter=False,
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
        self.emg_data = emg
        self.ref_data = ref[self.movement_name]
        # self.sample_lengths = [
        #     len(self.emg_data[i].shape[1]) for i in list(self.emg_data.keys())
        # ]
        self.sample_length = self.emg_data[self.movement_name].shape[2]
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
        self.use_gaussian_filter = use_gaussian_filter
        if self.use_gaussian_filter:
            self.gauss_filter = helpers.create_gaussian_filter(size_filter=5)

    def make_heatmap_emg(self, frame):
        """
        :return: heatmap of emg data

        """
        heatmap = helpers.calculate_heatmap(
            emg_grid=self.emg_data[self.movement_name], position=frame, interval_in_samples=self.number_observation_samples
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
            helpers.is_near_extremum(
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
            ) = helpers.check_to_which_movement_cycle_sample_belongs(
                frame, self.local_maxima, self.local_minima
            )
            # add the heatmap to the list of all heatmaps of the fitting flex/ex for later calculation of difference heatmap
            if belongs_to_movement == 1:
                # the closer the sample is to the extrema the more impact it has on the heatmap
                self.heatmaps_flex = np.add(
                    self.heatmaps_flex,
                    np.multiply(heatmap, 1 / (distance + 0.1)),
                )
                # add heatmap and not normalized heatmap because impact of all heatmaps will be the same but maybe some heatmaps have higher values and i want to use this ??
                # TODO maybe change this (better to use normalized or not ?)
                # IT IS BETER TO USE NORMALIZED BECAUSE IF EMG SCHWANKUNGEN
                self.number_heatmaps_flex += 1
            else:
                self.heatmaps_ex = np.add(
                    self.heatmaps_ex,
                    np.multiply(heatmap, 1 / (distance + 0.1)),
                )
                self.number_heatmaps_ex += 1

        self.global_counter += 1

    def get_heatmaps(self):
        """
        return the mean flex, mean ex and difference heatmap of one movement
        :return:
        """
        # samples = all sample values when using all samples with self.frame_duration in between

        self.samples =[i for i in range(0, self.sample_length, 64)]
        self.samples = [
            element for element in self.samples if element <= len(self.ref_data)
        ]
        # make both lists to save all coming heatmaps into it by adding the values and dividing at the end through number of heatmaps
        num_rows, num_cols,_ = self.emg_data[self.movement_name].shape
        self.heatmaps_flex = np.zeros((num_rows, num_cols))
        self.heatmaps_ex = np.zeros((num_rows, num_cols))
        self.number_heatmaps_flex = 0
        self.number_heatmaps_ex = 0
        if np.max(self.ref_data[:, 0]) > np.max(self.ref_data[:, 1]):
            self.local_maxima, self.local_minima = helpers.get_locations_of_all_maxima(
                self.ref_data[:, 0]
            )
            helpers.plot_local_maxima_minima(
                self.ref_data[:, 0], self.local_maxima, self.local_minima
            )
        else:
            self.local_maxima, self.local_minima = helpers.get_locations_of_all_maxima(
                self.ref_data[:, 1]
            )
            helpers.plot_local_maxima_minima(
                self.ref_data[:, 1], self.local_maxima, self.local_minima
            )

        for i in self.samples:
            self.make_heatmap_emg(i)

        heatmap_flex, heatmap_ex, heatmap_difference = self.heatmap_extraction()
        return heatmap_flex, heatmap_ex, heatmap_difference

    def get_channels(self):
        # samples = all sample values when using all samples with self.frame_duration in between
        self.samples =[i for i in range(0, self.sample_length, 64)]
        self.samples = [
            element for element in self.samples if element <= len(self.ref_data)
        ]
        # make both lists to save all coming heatmaps into it by adding the values and dividing at the end through number of heatmaps
        num_rows, num_cols,_ = self.emg_data[self.movement_name].shape
        self.heatmaps_flex = np.zeros((num_rows, num_cols))
        self.heatmaps_ex = np.zeros((num_rows, num_cols))
        self.number_heatmaps_flex = 0
        self.number_heatmaps_ex = 0


        if np.max(self.ref_data[:,0]) > np.max(self.ref_data[:, 1]):
            self.local_maxima, self.local_minima = helpers.get_locations_of_all_maxima(
                self.ref_data[:, 0], distance=5000
            )
            helpers.plot_local_maxima_minima(
                self.ref_data[:, 0], self.local_maxima, self.local_minima
            )
        else:
            self.local_maxima, self.local_minima = helpers.get_locations_of_all_maxima(
                self.ref_data[:, 1], distance=5000
            )
            helpers.plot_local_maxima_minima(
                self.ref_data[:, 1], self.local_maxima, self.local_minima
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

    def heatmap_extraction(self):
        mean_flex_heatmap = np.divide(self.heatmaps_flex, self.number_heatmaps_flex)
        if self.use_gaussian_filter:
            helpers.apply_gaussian_filter(mean_flex_heatmap, self.gauss_filter)

        mean_ex_heatmap = np.divide(self.heatmaps_ex, self.number_heatmaps_ex)
        if self.use_gaussian_filter:
            helpers.apply_gaussian_filter(mean_ex_heatmap, self.gauss_filter)

        mean_ex_heatmap[np.isnan(mean_ex_heatmap)] = 0
        mean_flex_heatmap[np.isnan(mean_flex_heatmap)] = 0
        difference_heatmap = np.subtract(mean_ex_heatmap, mean_flex_heatmap)
        difference_heatmap[np.isnan(difference_heatmap)] = 0

        return mean_flex_heatmap, mean_ex_heatmap, difference_heatmap

    def channel_extraction(self):
        mean_flex_heatmap = np.divide(self.heatmaps_flex, self.number_heatmaps_flex)

        mean_ex_heatmap = np.divide(self.heatmaps_ex, self.number_heatmaps_ex)

        mean_ex_heatmap[np.isnan(mean_ex_heatmap)] = 0
        mean_flex_heatmap[np.isnan(mean_flex_heatmap)] = 0
        difference_heatmap = np.subtract(mean_ex_heatmap, mean_flex_heatmap)
        difference_heatmap[np.isnan(difference_heatmap)] = 0

        channels_flexion, channels_extension = helpers.choose_possible_channels(
            difference_heatmap, mean_flex_heatmap, mean_ex_heatmap
        )
        print("choosen channels flexion: ", channels_flexion)
        print("choosen channels extension: ", channels_extension)
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

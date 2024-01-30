import numpy as np
from exo_controller import helpers
from exo_controller.grid_arrangement import Grid_Arrangement
from exo_controller.spatial_filters import Filters
import os


class Normalization:
    def __init__(
        self,
        method,
        grid_order,
        important_channels=None,
        sampling_frequency=2048,
        frame_duration=150,
        use_spatial_filter=False,
        use_muovi_pro = False,
    ):
        """
        :param movements: list of movements that should be used for normalization
        :param method: which method should be used for normalization -> "Min_Max_Scaling_all_channels","Min_Max_Scaling_over_whole_data","Robust_Scaling","Robust_all_channels"
        :param grid_order: list of the grid order i.e [1,2,3,4,5] or [1,2,3,5,4]
        :param important_channels: all channels that should be used as list -> [1,5,10] would only use these channels
        :param sampling_frequency: sampling frequency of the emg data
        :param frame_duration: duration of one frame in ms
        :return:
        """

        if important_channels is None:
            important_channels = range(320)
        self.important_channels = important_channels
        self.use_muovi_pro = use_muovi_pro
        self.grid_order = grid_order
        self.max_values = None
        self.min_values = None
        self.q1 = None
        self.q2 = None
        self.median = None
        self.mean = None
        self.all_emg_data = None
        self.method = method
        self.grid_aranger = Grid_Arrangement(grid_order,use_muovi_pro=use_muovi_pro)
        self.grid_aranger.make_grid()
        self.sampling_frequency = sampling_frequency
        self.frame_duration = frame_duration
        self.use_spatial_filter = use_spatial_filter
        if self.use_spatial_filter:
            self.spatial_filter = Filters()

    def robust_scaling(self, data):
        """
        scales the data with robust scaling
        :param data:
        :param q1:
        :param q2:
        :param median:
        :return:
        """
        if self.method == "Robust_Scaling":
            median = self.median[:, :, 0]
            q1 = self.q1[:, :, 0]
            q2 = self.q2[:, :, 0]
            return (np.array(data) - median) / ((q2 - q1)+ 1e-8)
        elif self.method == "Robust_all_channels":
            return (np.array(data) - self.median) / (self.q2 - self.q1)
        else:
            raise ValueError("Method not supported")

    def normalize_2D_array(self, data, axis=None, negative=False):
        """
        normalizes a 2D array

        :param data: the data to be normalized
        :param axis: which axis should be normalized, if None, the whole array will be normalized if 0 columns will be normalized, if 1 rows will be normalized
        :param negative: if True, the data will be normalized between -1 and 1, if False, the data will be normalized between 0 and 1
        :return:
        """
        data = np.array(data)
        if (self.max_values is not None) and (self.min_values is not None):
            if self.method == "Min_Max_Scaling_all_channels":
                min_value = self.min_values
                max_value = self.max_values
            else:
                min_value = self.min_values[:, :, 0]
                max_value = self.max_values[:, :, 0]

            norm = (data - min_value) / ((max_value - min_value)+1e-8)

        elif axis is None:
            data = np.array(data)
            norm = (data - np.min(data)) / ((np.max(data) - np.min(data))+ 1e-8)
        else:
            data = np.array(data)
            norm = (data - np.min(data, axis=axis)) / ((
                np.max(data, axis=axis) - np.min(data, axis=axis)
            ) + 1e-8 )

        if negative == True:
            norm = (norm * 2) - 1
        return norm

    def set_mean(self, mean):
        """
        sets the mean value that should be subtracted from the data
        :param mean: shape should be 320,1
        :return:
        """
        self.mean = mean

    def get_all_emg_data(self, path_to_data, movements):
        """
        returns all emg data in one array
        :param path_to_data: string with the path to the emg_data.pkl file where all the data is stored
        :param movements: list of movements that should be loaded
        :return:
        """
        all_emg_data = []
        a = helpers.load_pickle_file(path_to_data)
        print("movements", movements)

        for movement in movements:
            emg_data_one_movement = a[movement]#.transpose(1, 0, 2).reshape(64*len(self.grid_order), -1)


            if (self.mean is not None) and (self.mean is not None):
                # have to transfer self.mean_ex from grid arrangement to 320 channels arangement
                emg_data_one_movement = (
                    self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement(
                        emg_data_one_movement
                    )
                    - np.reshape(self.mean, (self.mean.shape[0], self.mean.shape[1], 1))
                )
            else:
                emg_data_one_movement = (
                    self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement(
                        emg_data_one_movement
                    )
                )

            # only take important channels and set all others to 0s
            if np.ndim(self.important_channels) == 2:
                emg_data_new = np.zeros((emg_data_one_movement.shape[0], emg_data_one_movement.shape[1],
                                         emg_data_one_movement.shape[2]))
                for important_channel in self.important_channels:
                    emg_data_new[important_channel[0], important_channel[1], :] = emg_data_one_movement[
                                                                                  important_channel[0],
                                                                                  important_channel[1], :]
                emg_data_one_movement = emg_data_new
            all_emg_data.append(emg_data_one_movement)

        self.all_emg_data = all_emg_data

    def calculate_norm_values_heatmap(self):
        """
        calculates the max min values to display all heatmaps in the same color range
        :return:
        """
        max_values = -np.inf
        min_values = np.inf
        for movement in range(len(self.all_emg_data)):
            sample_length = self.all_emg_data[movement].shape[2]
            num_samples = int(
                sample_length / (self.sampling_frequency * (self.frame_duration / 1000))
            )
            number_observation_samples = int(
                (self.frame_duration / 1000) * self.sampling_frequency
            )

            for frame in range(0, sample_length, number_observation_samples):
                heatmap = self.calculate_heatmap(
                    self.all_emg_data[movement], frame, number_observation_samples
                )
                if (self.mean is not None) and (self.mean is not None):
                    heatmap = heatmap - self.mean
                if np.max(heatmap) > max_values:
                    max_values = np.max(heatmap)
                if np.min(heatmap) < min_values:
                    min_values = np.min(heatmap)
        return max_values, min_values

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

    def calculate_heatmap_on_whole_samples(self, emg_grid):
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
                channel_data = emg_grid[row_idx][col_idx][:]
                # print(np.sqrt(np.mean(np.array(channel_data) ** 2)))
                rms_values[row_idx, col_idx] = np.sqrt(
                    np.mean(np.array(channel_data) ** 2)
                )
        return rms_values

    def set_method(self, method):
        """
        sets the method that should be used to normalize the data
        :param method: string with the method name
        :return:
        """
        self.method = method

    def calculate_normalization_values(self):
        """
        :param all_emg_data: list of all emg data for each movement -> each list entry is a array with shape (#rows,#cols,#samples)
        calculates all heatmaps for each movement and each sample and calculates the normalization values for the normalization method
        """

        if self.method == "no_scaling":
            return
        if self.method == "Min_Max_Scaling_all_channels":
            self.max_values = -np.inf
            self.min_values = np.inf

        if self.method == "Robust_Scaling":
            self.q1 = (
                np.zeros((self.all_emg_data[0].shape[0], self.all_emg_data[0].shape[1]))
                - np.inf
            )
            self.q2 = (
                np.zeros((self.all_emg_data[0].shape[0], self.all_emg_data[0].shape[1]))
                + np.inf
            )
            self.median = np.zeros(
                (self.all_emg_data[0].shape[0], self.all_emg_data[0].shape[1])
            )

        if self.method == "Robust_all_channels":
            self.q1 = -np.inf
            self.q2 = np.inf
            self.median = 0

        if self.method == "Min_Max_Scaling_over_whole_data":
            self.max_values = (
                np.zeros((self.all_emg_data[0].shape[0], self.all_emg_data[0].shape[1]))
                - np.inf
            )
            self.min_values = (
                np.zeros((self.all_emg_data[0].shape[0], self.all_emg_data[0].shape[1]))
                + np.inf
            )

        all_heatmaps = []

        for movement in range(len(self.all_emg_data)):


            sample_length = self.all_emg_data[movement][1][1].shape[0]
            num_samples = int(
                sample_length / (self.sampling_frequency * (self.frame_duration / 1000))
            )
            number_observation_samples = int(
                (self.frame_duration / 1000) * self.sampling_frequency
            )

            if self.use_muovi_pro:
                self.samples = [i for i in range(0, sample_length, 18)]
            else:
                self.samples =[i for i in range(0, sample_length, 64)]

            for frame in self.samples:
                if (frame - number_observation_samples < 0) or (
                        self.all_emg_data[movement].shape[2] < (number_observation_samples)
                ):
                    emg_data_to_use = self.all_emg_data[movement][:,:,: frame + 1]
                else:
                    emg_data_to_use = self.all_emg_data[movement][:,:,
                                   frame - number_observation_samples: frame
                                   ]


                if self.use_spatial_filter:
                    emg_data_to_use = self.spatial_filter.spatial_filtering(emg_data_to_use,  filter_name="IR")

                heatmap = self.calculate_heatmap_on_whole_samples(
                    emg_data_to_use
                )



                if (self.mean is not None) and (self.mean is not None):
                    heatmap = np.subtract(heatmap ,self.mean)

                if self.method == "Min_Max_Scaling_all_channels":
                    if np.max(heatmap) > self.max_values:
                        self.max_values = np.max(heatmap)
                    if np.min(heatmap) < self.min_values:
                        self.min_values = np.min(heatmap)

                if self.method == "Min_Max_Scaling_over_whole_data":
                    for row in range(heatmap.shape[0]):
                        for col in range(heatmap.shape[1]):
                            if heatmap[row, col] > self.max_values[row, col]:
                                self.max_values[row, col] = heatmap[row, col]
                            if heatmap[row, col] < self.min_values[row, col]:
                                self.min_values[row, col] = heatmap[row, col]

                if "Robust" in self.method:
                    all_heatmaps.append(heatmap)

        if self.method == "Min_Max_Scaling_over_whole_data":
            self.max_values = np.reshape(
                self.max_values, (self.max_values.shape[0], self.max_values.shape[1], 1)
            )
            self.min_values = np.reshape(
                self.min_values, (self.min_values.shape[0], self.min_values.shape[1], 1)
            )

        if self.method == "Robust_Scaling":
            all_heatmaps = np.stack(np.array(all_heatmaps), axis=-1)
            self.q1 = np.quantile(all_heatmaps, 0.25, axis=-1)
            self.q2 = np.quantile(all_heatmaps, 0.75, axis=-1)
            self.median = np.median(all_heatmaps, axis=-1)

            self.q1 = np.reshape(self.q1, (self.q1.shape[0], self.q1.shape[1], 1))
            self.q2 = np.reshape(self.q2, (self.q2.shape[0], self.q2.shape[1], 1))
            self.median = np.reshape(
                self.median, (self.median.shape[0], self.median.shape[1], 1)
            )

        if self.method == "Robust_all_channels":
            all_heatmaps = np.stack(np.array(all_heatmaps), axis=-1)
            self.q1 = np.min(np.quantile(all_heatmaps, 0.25, axis=-1))
            self.q2 = np.max(np.quantile(all_heatmaps, 0.75, axis=-1))
            self.median = np.mean(np.median(all_heatmaps, axis=-1))

    def normalize_chunk(self, chunk):
        """
        normalizes a chunk of data
        :param chunk: chunk of data that should be normalized
        :return:
        """

        if self.method == "Min_Max_Scaling_all_channels":
            return self.normalize_2D_array(chunk)
        elif self.method == "Min_Max_Scaling_over_whole_data":
            return self.normalize_2D_array(chunk)
        elif self.method == "Robust_Scaling":
            return self.robust_scaling(chunk)
        elif self.method == "Robust_all_channels":
            return self.robust_scaling(chunk)
        elif self.method == "no_scaling":
            return chunk
        else:
            raise ValueError("Method not supported")

#
# if __name__ == "__main__":
#     pass
    # workflow
    # normalizer = Normalization()
    # normalizer.get_all_emg_data(path_to_data="C:/Users/Philipp/Desktop/BA/exo_controller/data/emg_data.pkl",movements=["flexion","extension","pronation","supination","hand_close","hand_open"])
    # normalizer.find_max_min_values_for_each_movement_and_channel(emg_data=normalizer.all_emg_data)
    # normalizer.find_q_median_values_for_each_movement_and_channel(emg_data=normalizer.all_emg_data)
    # normalizer.set_mean(channel_extraction_get_mean)
    #
    # heatmap = get heatmap
    # heatmap = heatmap - normalizer.mean
    # heatmap = normalizer.normalize_2D_array(emg_data)

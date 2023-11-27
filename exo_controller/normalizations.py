import numpy as np
from exo_controller.helpers import load_pickle_file
import os


class Normalization:
    def __inti__(self,movements,important_channels=None):
        self.movements = movements
        if important_channels is None:
            important_channels = range(320)
        self.important_channels = important_channels
        self.max_values = None
        self.min_values = None
        self.q1 = None
        self.q2 = None
        self.median = None
        self.mean = None
        self.all_emg_data = None

    def calculate_normalization_values(self,method):
        """
        calculates the normalization values
        :param method: which method should be used to calculate the normalization values
        :return:
        """
        # "Min_Max_Scaling_over_whole_data" = min max scaling with max/min is choosen individually for every channel
        # "Robust_all_channels" = robust scaling with q1,q2,median is choosen over all channels
        # "Robust_Scaling"  = robust scaling with q1,q2,median is choosen individually for every channel
        # "Min_Max_Scaling_all_channels" = min max scaling with max/min is choosen over all channels


        if method == "Min_Max_Scaling_over_whole_data":
            return self.find_max_min_values_for_each_movement_and_channel(emg_data=self.all_emg_data, important_channels=self.important_channels)
        elif method == "Robust_all_channels":
            return self.find_q_median_values_for_each_movement(emg_data=self.all_emg_data, important_channels=self.important_channels)
        elif method == "Robust_Scaling":
            return self.find_q_median_values_for_each_movement_and_channel(emg_data=self.all_emg_data, important_channels=self.important_channels)
        elif method == "Min_Max_Scaling_all_channels":
            return self.find_max_min_values_over_all_movements(emg_data=self.all_emg_data, important_channels=self.important_channels)
        elif method == "no_scaling":
            return None
        else:
            raise ValueError("method not supported")

    def find_max_min_values_for_each_movement_and_channel(self,emg_data, important_channels=None):
        """
        This function finds the max and min values for each channel individually in the emg data.
        It will return the max and min values for each channel over all movements.
        This is needed to normalize the data later on.
        :param emg_data: the emg data in the form of a list of movements, each movement is a 2D array with the shape (num_channels,num_samples)
        :return: max and min values for each channel in the form of two lists (max /min) with each having 320 entries (one for each channel)
        """
        if important_channels is None:
            important_channels = self.important_channels

        max_values = np.zeros(len(important_channels)) - 10000000
        min_values = np.zeros(len(important_channels)) + 10000000
        for movement in range(len(emg_data)):
            for channel in important_channels:
                if np.max(emg_data[movement][channel]) > max_values[channel]:
                    max_values[channel] = np.max(emg_data[movement][channel])
                if np.min(emg_data[movement][channel]) < min_values[channel]:
                    min_values[channel] = np.min(emg_data[movement][channel])
        self.max_values = np.array(max_values)
        self.min_values = np.array(min_values)
        return max_values, min_values



    def find_max_min_values_over_all_movements(self,emg_data, important_channels, movements):
        """
        This function finds the max and min values in the emg data but not channel wise but rather max min over all channels and movements.
        It will return the max and min values for each channel over all movements.
        This is needed to normalize the data later on.
        :return: max and min values in the form of one value for max and min that is the highest and lowest value over all channels and movements
        """
        max_values = -10000000
        min_values = 10000000
        # for movement in movements:
        for movement in range(len(emg_data)):
            for channel in important_channels:
                if np.max(emg_data[movement][channel]) > max_values:
                    max_values = np.max(emg_data[movement][channel])
                if np.min(emg_data[movement][channel]) < min_values:
                    min_values = np.min(emg_data[movement][channel])
        self.max_values = np.array(max_values)
        self.min_values = np.array(min_values)
        return max_values, min_values

    def find_q_median_values_for_each_movement(self,emg_data, important_channels):
        """
        This function finds the q1 and q2 values over all movements and channels in the emg data.
        It will return the max and min values for each channel over all movements.
        This is needed to normalize the data later on.
        :return: q1,q2,median vlaues  -> 3 values which are over alll channels and movements
        """
        # TODO wie will ich das hier machen ?? welche median und quantiles nehme ich ??  ich sollte eigentlich genau so machen (größtes 1 und 3 quantile) median keine ahnung :D
        q1 = None
        q2 = None
        median = None

        count = 0
        # for movement in movements:
        for movement in range(len(emg_data)):
            for channel in important_channels:
                if median is None:
                    median = np.median(emg_data[movement][channel])
                else:
                    median += np.median(emg_data[movement][channel])
                count += 1
                if q1 is None:
                    q1 = np.quantile(emg_data[movement][channel], 0.25)
                if q2 is None:
                    q2 = np.quantile(emg_data[movement][channel], 0.75)
                if np.quantile(emg_data[movement][channel], 0.25) < q1:
                    q1 = np.quantile(emg_data[movement][channel], 0.25)
                if np.quantile(emg_data[movement][channel], 0.75) > q2:
                    q2 = np.quantile(emg_data[movement][channel], 0.75)

        for i in range(len(important_channels)):
            median = median / count
        self.q1 = np.array(q1)
        self.q2 = np.array(q2)
        self.median = np.array(median)
        return q1, q2, median

    def find_q_median_values_for_each_movement_and_channel(self,emg_data, important_channels):
        """
        This function finds the q1,q2,median values for each channel individually in the emg data.
        It will return the max and min values for each channel over all movements.
        This is needed to normalize the data later on.
        :return: q1,q2,median values for each channel
        """
        # TODO wie will ich das hier machen ?? welche median und quantiles nehme ich ??  ich sollte eigentlich genau so machen (größtes 1 und 3 quantile) median keine ahnung :D
        q1 = np.zeros(len(important_channels)) - 10000000
        q2 = np.zeros(len(important_channels)) + 10000000
        median = np.zeros(len(important_channels))

        for movement in range(len(emg_data)):
            # for movement in movements:
            for channel in important_channels:
                median[channel] += np.median(emg_data[movement][channel])
                if np.quantile(emg_data[movement][channel], 0.25) < q1[channel]:
                    q1[channel] = np.quantile(emg_data[movement][channel], 0.25)
                if np.quantile(emg_data[movement][channel], 0.75) > q2[channel]:
                    q2[channel] = np.quantile(emg_data[movement][channel], 0.75)

        for i in range(len(important_channels)):
            median[i] = median[i] / len(emg_data)
        self.q1 = np.array(q1)
        self.q2 = np.array(q2)
        self.median = np.array(median)
        return q1, q2, median

    def robust_scaling(self,data):
        """
        scales the data with robust scaling
        :param data:
        :param q1:
        :param q2:
        :param median:
        :return:
        """
        return (np.array(data) - self.median) / (self.q2 - self.q1)

    def normalize_2D_array(self,data, axis=None, negative=False):
        """
        normalizes a 2D array

        :param data: the data to be normalized
        :param axis: which axis should be normalized, if None, the whole array will be normalized if 0 columns will be normalized, if 1 rows will be normalized
        :param negative: if True, the data will be normalized between -1 and 1, if False, the data will be normalized between 0 and 1
        :return:
        """
        data = np.array(data)
        if (self.max_values is not None) and (self.min_values is not None):
            min_value = self.min_values
            max_value = self.max_values
            norm = (data - min_value) / ((max_value - min_value))

        elif axis is None:
            data = np.array(data)
            norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        else:
            data = np.array(data)
            norm = (data - np.min(data, axis=axis)) / (np.max(data, axis=axis) - np.min(data, axis=axis))

        if negative == True:
            norm = (norm * 2) - 1
        return norm

    def set_mean(self,mean):
        """
        sets the mean value that should be subtracted from the data
        :param mean: shape should be 320,1
        :return:
        """
        self.mean = mean

    def get_all_emg_data(self,path_to_data,movements):
        """
        returns all emg data in one array
        :param path_to_data: string with the path to the emg_data.pkl file where all the data is stored
        :param movements: list of movements that should be loaded
        :return:
        """
        all_emg_data = []
        a = load_pickle_file(path_to_data)

        for movement in movements:
            emg_data_one_movement = a[movement].transpose(1, 0, 2).reshape(320, -1)

            if self.mean is not None:
                # have to transfer self.mean_ex from grid arrangement to 320 channels arangement
                emg_data_one_movement = emg_data_one_movement - self.mean
            all_emg_data.append(emg_data_one_movement)

        self.all_emg_data = all_emg_data


if __name__ == "__main__":
    pass
    #workflow
    # normalizer = Normalization()
    # normalizer.get_all_emg_data(path_to_data="C:/Users/Philipp/Desktop/BA/exo_controller/data/emg_data.pkl",movements=["flexion","extension","pronation","supination","hand_close","hand_open"])
    # normalizer.find_max_min_values_for_each_movement_and_channel(emg_data=normalizer.all_emg_data)
    # normalizer.find_q_median_values_for_each_movement_and_channel(emg_data=normalizer.all_emg_data)
    # normalizer.set_mean(channel_extraction_get_mean)
    #
    # heatmap = get heatmap
    # heatmap = heatmap - normalizer.mean
    # heatmap = normalizer.normalize_2D_array(emg_data)


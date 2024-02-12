import numpy as np
from scipy.signal import fftconvolve
from exo_controller import helpers
from scipy.signal import butter, lfilter, iirnotch


class Filters:
    def __init__(self):
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

    def spatial_filtering(self, siganl_to_be_filtered, filter_name="NDD", mode="same"):
        """
        :param siganl_to_be_filtered: in shape (rows,cols,length emg signal) = grid
        :param filter_name: name of the filter to be applied i.e identity, LSD, LDD, TSD, TDD, NDD, IB2, IR
        :return: the filtered grid
        """

        grid = siganl_to_be_filtered
        grids = helpers.split_grid_into_8x8_grids(grid)
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
                    mode=mode,
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
                    mode=mode,
                ).astype(
                    np.float32
                )

        return grid

    def create_gaussian_filter(self, size_filter=3, sigma=None):
        """
        Creates a Gaussian filter of the given size and standard deviation.

        :param size: The size of the filter (size x size)
        :param sigma: The standard deviation of the Gaussian filter. If None, sigma is set to size/6.
        :return: A 2D numpy array representing the Gaussian filter
        """
        if sigma is None:
            sigma = (
                size_filter / 6.0
            )  # A rule of thumb is to set sigma to 1/6 of the kernel size

        kernel = np.ones((size_filter, size_filter))
        for row in range(size_filter):
            for col in range(size_filter):
                x = col - size_filter // 2
                y = row - size_filter // 2
                kernel[row, col] = np.multiply(
                    np.divide(1, 2 * np.pi * sigma**2),
                    np.exp(-(x**2 + y**2) / (2 * sigma**2)),
                )

        # Normalize the kernel to ensure the sum of all elements is 1
        kernel /= np.sum(kernel)
        self.kernel = kernel

        return kernel

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        bandpass filter the data
        :param data:
        :param lowcut:
        :param highcut:
        :param fs:
        :param order:
        :return:
        """
        order = 5
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b_notch, a_notch = iirnotch(50, 30, fs)
        b, a = butter(order, [low, high], btype="band", fs=fs, analog=False)
        y = lfilter(b_notch, a_notch, data)
        y = lfilter(b, a, data)
        return y

    def bandpass_filter_emg_data(self, emg_data, fs=2048):
        """
        bandpass filter the emg data
        :param emg_data: in shape 320, length emg signal
        :param fs: sampling freq of the emg signal
        :return:
        """
        data = np.copy(emg_data)
        channel_filter_instances = []
        for channel in range(emg_data.shape[0]):
            data[channel] = self.bandpass_filter(
                emg_data[channel], 10, 500, fs, order=5
            )
        return data

    def bandpass_filter_grid_emg_data(self, emg_data, fs=2048):
        """
        bandpass filter the emg data
        :param emg_data: in shape 320, length emg signal
        :param fs: sampling freq of the emg signal
        :return:
        """
        data = np.copy(emg_data)

        for row in range(emg_data.shape[0]):
            for col in range(emg_data.shape[1]):
                data[row, col] = self.bandpass_filter(
                    emg_data[row, col], 10, 500, fs, order=5
                )
        return data

    def apply_gaussian_filter(self, grid, gaussian_filter):
        """
        Applies the Gaussian filter to the grid to smooth the values.

        :param grid: A rows x cols numpy array representing the grid
        :param gaussian_filter: A 2D numpy array representing the Gaussian filter
        :return: A rows x cols numpy array with smoothed values
        """

        # Define the function to handle the border
        grid = grid.reshape(grid.shape[0], grid.shape[1])
        grids = helpers.split_grid_into_8x8_grids(
            grid.reshape(grid.shape[0], grid.shape[1])
        )

        def filter_function(image, filter):
            filtered_image = np.zeros_like(image)
            filter_size = filter.shape[0]
            pad_width = filter_size // 2

            # Pad the image with the edge values to handle borders
            padded_image = np.pad(image, pad_width, mode="edge")

            # Iterate over each cell in the original image
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    # Define the region of interest in the padded image
                    region = padded_image[i : i + filter_size, j : j + filter_size]
                    # Apply the filter, ignoring the parts that go beyond the borders
                    filtered_value = np.sum(region * filter)
                    filtered_image[i, j] = filtered_value

            return filtered_image

        for one_grid in range(len(grids)):
            if one_grid < 3:  # if upper grids
                grid[0:8, 0 + 8 * one_grid : 8 + 8 * one_grid] = filter_function(
                    grids[one_grid], gaussian_filter
                )
            else:  # if lower grids
                grid[
                    8:16, 0 + 8 * (one_grid - 3) : 8 + 8 * (one_grid - 3)
                ] = filter_function(grids[one_grid], gaussian_filter)

        return grid

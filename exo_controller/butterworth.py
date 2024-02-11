from __future__ import annotations

from typing import TYPE_CHECKING, Union
import numpy as np
from scipy.signal import butter, lfilter, iirnotch

if TYPE_CHECKING:
    pass


class Bandpass:
    def __init__(
        self,
        lowcut: Union[int, float],
        highcut: Union[int, float],
        fs: Union[int, float],
        order: int = 2,
    ) -> None:
        """
        Initialize the real-time Bandpass filter.

        Args:
            lowcut (int, float): The low cutoff frequency in Hz.
            highcut (int, float): The high cutoff frequency in Hz.
            fs (int, float): The sampling frequency in Hz.
            order (int): The filter order (default is 2).
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

        # Create Butterworth bandpass filter coefficients
        self.lowpass = Lowpass(cutoff=highcut, fs=self.fs, order=order)
        self.highpass = Highpass(cutoff=lowcut, fs=self.fs, order=order)

    def filter_channel(
        self, x: Union[float, np.ndarray, list], multiple_samples: bool = False
    ) -> Union[float, np.ndarray, list]:
        """
        Filter a single input sample.

        Args:
            x (float, np.ndarray, list): The input sample to be filtered.
            single_channel (bool): Whether the input is a single sample or a list of samples.

        Returns:
            float, np.ndarray, list: The filtered output sample(s).
        """
        # Apply the filters
        temp = self.lowpass.filter_channel(x, multiple_samples=multiple_samples)
        return self.highpass.filter_channel(temp, multiple_samples=multiple_samples)


class Lowpass:
    def __init__(
        self, cutoff: Union[int, float], fs: Union[int, float], order: int = 2
    ) -> None:
        """
        Initialize the real-time lowpass filter.

        Args:
            cutoff (int, float): The cutoff frequency in Hz.
            fs (int, float): The sampling frequency in Hz.
            order (int): The filter order (default is 2).
        """
        self.cutoff = cutoff
        self.fs = fs
        self.order = order

        # Compute the Nyquist frequency
        nyquist = 0.5 * fs

        # Calculate the Nyquist normalized cutoff frequency
        self.cutoff_norm = cutoff / nyquist

        # Create Butterworth lowpass filter coefficients
        self.b, self.a = butter(order, self.cutoff_norm, btype="low")

        # Initialize the filter state
        self.z = np.zeros(order)

    def filter_channel(
        self, x: Union[float, np.ndarray, list], multiple_samples: bool = False
    ) -> Union[float, np.ndarray, list]:
        """
        Filter a single input sample.

        Args:
            x (float, np.ndarray, list): The input sample to be filtered.
            single_channel (bool): Whether the input is a single sample or a list of samples.

        Returns:
            (float, np.ndarray, list): The filtered output sample(s).
        """
        # Apply the filter
        if not multiple_samples:
            y, self.z = lfilter(self.b, self.a, [x], zi=self.z)
            return y[0]
        else:
            y, self.z = lfilter(self.b, self.a, x, zi=self.z)
            return y


class Highpass:
    def __init__(
        self, cutoff: Union[int, float], fs: Union[int, float], order: int = 2
    ) -> None:
        """
        Initialize the real-time highpass filter.

        Args:
            cutoff (int, float): The cutoff frequency in Hz.
            fs (int, float): The sampling frequency in Hz.
            order (int): The filter order (default is 2).
        """
        self.cutoff = cutoff
        self.fs = fs
        self.order = order

        # Compute the Nyquist frequency
        nyquist = 0.5 * fs

        # Calculate the Nyquist normalized cutoff frequency
        self.cutoff_norm = cutoff / nyquist

        # Create Butterworth highpass filter coefficients
        self.b, self.a = butter(order, self.cutoff_norm, btype="high")

        # Initialize the filter state
        self.z = np.zeros(order)

    def filter_channel(
        self, x: Union[float, np.ndarray, list], multiple_samples: bool = False
    ) -> Union[float, np.ndarray, list]:
        """
        Filter a single input sample.

        Args:
            x (float, np.ndarray, list): The input sample to be filtered.
            single_channel (bool): Whether the input is a single sample or a list of samples.

        Returns:
            (float, np.ndarray, list): The filtered output sample(s).
        """
        # Apply the filter
        if not multiple_samples:
            y, self.z = lfilter(self.b, self.a, [x], zi=self.z)
            return y[0]
        else:
            y, self.z = lfilter(self.b, self.a, x, zi=self.z)
            return y


class Notch:
    def __init__(
        self,
        center_freq: Union[int, float],
        fs: Union[int, float],
        q: float = 30,
    ) -> None:
        """
        Initialize the real-time Notch filter.

        Args:
            center_freq (int, float): The center frequency of the notch filter in Hz.
            fs (int, float): The sampling frequency in Hz.
            q (float): The quality factor of the notch filter.
        """
        self.center_freq = center_freq
        self.q = q
        self.fs = fs

        # Calculate the notch filter coefficients
        self.b, self.a = iirnotch(center_freq, q, fs=fs)

        # Initialize the filter state
        self.z = [0.0] * (len(self.b) - 1)

    def filter_channel(
        self, x: Union[float, np.ndarray, list], multiple_samples: bool = False
    ) -> Union[float, np.ndarray, list]:
        """
        Filter a single input sample.

        Args:
            x (float, np.ndarray, list): The input sample to be filtered.
            single_channel (bool): Whether the input is a single sample or a list of samples.

        Returns:
            (float, np.ndarray, list): The filtered output sample(s).
        """
        # Apply the filter
        if not multiple_samples:
            y, self.z = lfilter(self.b, self.a, [x], zi=self.z)
            return y[0]
        else:
            y, self.z = lfilter(self.b, self.a, x, zi=self.z)
            return y

from typing import List, Tuple, Literal

import numpy as np
from scipy.signal import fftconvolve

# Dictionary below is used to define differential filters that can be applied across the monopolar electrode grids
_DIFFERENTIAL_FILTERS = {
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
    "IR": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),  # inverse rectangle
}

# Dictionary below is used to define signal processing steps that need to be taken in order to convert the original
# monopolar electrode recording system into the setup used in another experiment
ELECTRODE_SETUP = {
    "Quattrocento_full": {
        "grid": ("GR10MM0808", (5, 8, 8), list(np.arange(320))),
        "concatenate": False,
        "average": (1, "longitudinal"),
        "differential": "identity",
        "channel_selection": None,
    },
    "Quattrocento_forearm": {
        "grid": ("GR10MM0808", (3, 8, 8), list(np.arange(64, 256))),
        "concatenate": False,
        "average": (1, "longitudinal"),
        "differential": "identity",
        "channel_selection": None,
    },
    "Quattrocento_forearm_reduced": {
        "grid": ("GR10MM0808", (3, 8, 8), list(np.arange(64, 256))),
        "concatenate": False,
        "average": (1, "longitudinal"),
        "differential": "IR",
        "channel_selection": None,
    },
    "Quattrocento_forearm_ablated": {
        "grid": ("GR10MM0808", (3, 8, 8), list(np.arange(64, 256))),
        "concatenate": True,
        "average": (1, "longitudinal"),
        "differential": "identity",
        "channel_selection": [
            (3, 2),
            (3, 5),
            (3, 8),
            (3, 11),
            (3, 14),
            (3, 17),
            (3, 20),
            (3, 23),
            (4, 2),
            (4, 5),
            (4, 8),
            (4, 11),
            (4, 14),
            (4, 17),
            (4, 20),
            (4, 23),
        ],
    },
    "Myobock": {
        "grid": ("GR10MM0808", (3, 8, 8), list(np.arange(64, 256))),
        "concatenate": False,
        "average": (2, "transverse"),
        "differential": "LDD",
        "channel_selection": [(2, 0), (2, 2), (2, 4), (2, 6)],
    },
    "Thalmic_Myoarmband": {
        "grid": ("GR10MM0808", (3, 8, 8), list(np.arange(64, 256))),
        "concatenate": True,
        "average": (2, "transverse"),
        "differential": "LSD",
        "channel_selection": [
            (0, 0),
            (0, 3),
            (0, 6),
            (0, 9),
            (0, 12),
            (0, 15),
            (0, 18),
            (0, 21),
            (6, 1),
            (6, 4),
            (6, 7),
            (6, 10),
            (6, 13),
            (6, 16),
            (6, 19),
            (6, 22),
        ],
    },
}


class GridReshaping:
    def __init__(
        self,
        chunk: np.ndarray,
        nr_grids: int,
        nr_rows: int,
        nr_col: int,
        grid_type: str,
    ):
        """Initialize the class.

        Attributes
        ----------
        chunk : np.ndarray
            Input array to reshape.
        nr_grids : int
            Nr of electrode grids included in the whole channel list.
        nr_rows : int
            Nr of rows that the electrode grid has or should have after reshaping.
        nr_col : int
            Nr of columns that the electrode grid has or should have after reshaping.
        grid_type : str
            Type of grid to be reshaped. Either 8x8 grid of 10mm IED or a 13x5 grid of 8mm IED.
            Can be either "GR10MM0808" or "GR08MM1305".
        """
        self.chunk = chunk
        self.nr_grids = nr_grids
        self.nr_rows = nr_rows
        self.nr_col = nr_col
        self.grid_type = grid_type

    def channels_to_grid(self):
        """Reshape input chunk to grid shape. Use this function before any spatial filtering to avoid reshaping errors."""
        nr_filter_representations = self.chunk.shape[0]
        if self.grid_type == "GR10MM0808":
            self.chunk = self.chunk.reshape(
                (nr_filter_representations, self.nr_grids, 64, -1)
            ).reshape((nr_filter_representations, self.nr_grids, 8, 8, -1), order="F")[
                :, :, ::-1
            ]

            return self.chunk
        elif self.grid_type == "GR08MM1305":
            self.chunk = np.pad(
                self.chunk.reshape((nr_filter_representations, self.nr_grids, 64, -1)),
                ((0, 0), (0, 0), (1, 0), (0, 0)),
                "constant",
            ).reshape((nr_filter_representations, self.nr_grids, 13, 5, -1), order="F")
            self.chunk[:, :, :, [1, 3]] = np.flip(self.chunk[:, :, :, [1, 3]], axis=2)

            return self.chunk

        else:
            raise ValueError("This electrode grid is not defined.")

    def grid_to_channels(self):
        """Reshape input chunk with the electrode grid shape back to channel x samples format. Use this function after
        applying spatial filters.
        """
        nr_filter_representations = self.chunk.shape[0]
        if self.grid_type == "GR10MM0808":
            chunk = self.chunk[:, :, ::-1]
            orig_chunk = chunk[:, 0].reshape(
                (nr_filter_representations, self.nr_rows * self.nr_col, -1), order="F"
            )

            for i in range(1, chunk.shape[1]):
                orig_chunk = np.concatenate(
                    (
                        orig_chunk,
                        chunk[:, i].reshape(
                            (nr_filter_representations, self.nr_rows * self.nr_col, -1),
                            order="F",
                        ),
                    ),
                    axis=1,
                )

            self.chunk = orig_chunk
            return self.chunk
        elif self.grid_type == "GR08MM1305":
            chunk = self.chunk
            chunk[:, :, :, 1] = np.flip(chunk[:, :, :, 1], axis=2)

            if 5 - self.nr_col < 2:
                chunk[:, :, :, 3] = np.flip(chunk[:, :, :, 3], axis=2)

            orig_chunk = chunk[:, 0].reshape(
                (nr_filter_representations, self.nr_rows * self.nr_col, -1), order="F"
            )

            for i in range(1, chunk.shape[1]):
                orig_chunk = np.concatenate(
                    (
                        orig_chunk,
                        chunk[:, i].reshape(
                            (nr_filter_representations, self.nr_rows * self.nr_col, -1),
                            order="F",
                        ),
                    ),
                    axis=1,
                )

            self.chunk = orig_chunk
            return self.chunk

        else:
            raise ValueError("This electrode grid is not defined.")

    def grid_concatenation(self):
        """Concatenate all electrode grids along specified axis. Function can be used to apply a spatial filter
        across multiple arrays, for e.g., in the direction of the arm circumference.
        """
        concatenated_grid = self.chunk[:, 0]

        for i in range(1, self.chunk.shape[1]):
            concatenated_grid = np.concatenate(
                (concatenated_grid, self.chunk[:, i]), axis=-2
            )

        self.chunk = concatenated_grid
        return np.expand_dims(self.chunk, axis=1)


class DifferentialSpatialFilter:
    def __init__(self, chunk: np.ndarray, filter_name: str):
        """Initialize the class.

        Attributes
        ----------
        chunk : np.ndarray
            Input EMG array with grid shape to be filtered.
        filter_name : str
            Name of the filter to be applied: "LSD", "TSD", "LDD", "TDD", "NDD", "IB2" or "IR". Filters are defined
            according to https://doi.org/10.1109/TBME.2003.808830. In case no filter is applied, use "identity".
        """
        self.chunk = chunk
        self.filter_name = filter_name

    def spatial_filtering(self, siganl_to_be_filtered, filter_name):
        """This function applies the filters to the chunk."""

        # Extend filter dimensions prior to performing a convolution
        flt_coeff = np.expand_dims(_DIFFERENTIAL_FILTERS[filter_name], axis=(0, 1, -1))
        filtered_chunk = fftconvolve(
            siganl_to_be_filtered, flt_coeff, mode="valid"
        ).astype(np.float32)

        self.chunk = filtered_chunk
        return self.chunk


class AveragingSpatialFilter:
    def __init__(self, chunk: np.ndarray, order: int, filter_direction: str):
        """Initialize the class.

        Attributes
        ----------
        chunk : np.ndarray
            Input EMG array with grid shape to be filtered.
        order : int
            Order of the moving average filter.
        filter_direction : str
            Grid direction over which the filter is applied. Can be either "longitudinal" or "transverse".
        """
        self.chunk = chunk
        self.order = order
        self.filter_direction = filter_direction

    def moving_avg(self):
        """This function applies the moving average filter across the chunk."""

        if self.filter_direction == "longitudinal":
            flt_coeff = np.expand_dims(
                1
                / self.order
                * np.ones(self.order, dtype=int).reshape((self.order, -1)),
                axis=(0, 1, -1),
            )
        elif self.filter_direction == "transverse":
            flt_coeff = np.expand_dims(
                1
                / self.order
                * np.ones(self.order, dtype=int).reshape((-1, self.order)),
                axis=(0, 1, -1),
            )
        else:
            raise ValueError("Averaging direction name not correct.")

        # Extend filter dimensions prior to performing a convolution
        filtered_chunk = fftconvolve(self.chunk, flt_coeff, mode="valid").astype(
            np.float32
        )

        self.chunk = filtered_chunk
        return self.chunk


class ChannelSelection:
    def __init__(self, chunk: np.ndarray, grid_position):
        """Initialize the class.

        Attributes
        ----------
        chunk : np.ndarray
            Input grid array.
        grid_position : List[Tuple[int, int]]
            List of all grid electrode indexes based on row-column combination. If no channel selection is performed,
            set to None.
        """
        self.chunk = chunk
        self.grid_position = grid_position

    def select_channel(self):
        """Select desired channels from each grid. If grids have not been concatenated, then the provided list of
        channel indexes apply the same to each grid."""
        if self.grid_position is None:
            return self.chunk
        else:
            selected_channel = []

            for index in self.grid_position:
                selected_channel.append(self.chunk[:, :, index[0], index[1]])

            self.chunk = np.array(selected_channel)
            self.chunk = self.chunk.reshape((1, -1, self.chunk.shape[-1]), order="F")

            return self.chunk


if __name__ == "__main__":
    # Test the classes out
    # All the non-grid reshaping functions below should be applied to chunks of shape grid x row x col x samples and
    # not to chunks of shape channels x samples

    emg_data = np.random.rand(1, 320, 192)
    test_emg = emg_data
    # emg_setup = "Thalmic-Myoarmband"
    emg_setup = "Quattrocento-forearm"

    print(
        "EMG dataset shape prior to any spatial filtering and reshaping:",
        emg_data.shape,
    )

    emg_data = emg_data[:, ELECTRODE_SETUP[emg_setup]["grid"][-1]]
    grid_reshaper = GridReshaping(
        chunk=emg_data,
        nr_grids=ELECTRODE_SETUP[emg_setup]["grid"][1][0],
        nr_rows=ELECTRODE_SETUP[emg_setup]["grid"][1][1],
        nr_col=ELECTRODE_SETUP[emg_setup]["grid"][1][2],
        grid_type=ELECTRODE_SETUP[emg_setup]["grid"][0],
    )
    emg_data = grid_reshaper.channels_to_grid()

    print(
        "EMG dataset shape after reshaping from 1D channels array to grid shape",
        emg_data.shape,
    )

    if ELECTRODE_SETUP[emg_setup]["concatenate"]:
        emg_data = grid_reshaper.grid_concatenation()

    print("EMG dataset shape after concatenating all grids together", emg_data.shape)

    emg_data = AveragingSpatialFilter(
        chunk=emg_data,
        order=ELECTRODE_SETUP[emg_setup]["average"][0],
        filter_direction=ELECTRODE_SETUP[emg_setup]["average"][1],
    ).moving_avg()

    print("EMG dataset shape after applying the averaging filter", emg_data.shape)

    emg_data = DifferentialSpatialFilter(
        chunk=emg_data, filter_name=ELECTRODE_SETUP[emg_setup]["differential"]
    ).spatial_filtering()

    print("EMG dataset shape after applying the differential filter", emg_data.shape)

    if ELECTRODE_SETUP[emg_setup]["channel_selection"] is not None:
        emg_data = ChannelSelection(
            chunk=emg_data,
            grid_position=ELECTRODE_SETUP[emg_setup]["channel_selection"],
        ).select_channel()
    else:
        emg_data = grid_reshaper.grid_to_channels()

    print(
        "EMG dataset shape after selecting the needed channels and returning to 1D array shape",
        emg_data.shape,
    )

import numpy as np


class Grid_Arrangement:
    def __init__(self, channel_order=None, use_muovi_pro=False):
        self.num_grids = len(channel_order)
        self.channel_order = channel_order  # which grid is placed on which quattrocento outlet [2,1,4,5,3] means outlet 2 is the first grid, besides is outlet 1 as second grid...
        self.use_muovi_pro = use_muovi_pro
    def make_grid(self):
        """
        creates the grid arrangement for the quattrocento it creates a self variable upper_grid and lower_grid if 5 grids are used otherwise only upper_grid , upper grid = 3 grids besides , lower grid = 2 besides, furthermore it uses the way how we used the outlets from the quattrocento to fit it to the grid positions
        :return:
        """
        if self.use_muovi_pro:
            array = np.reshape(np.arange(1,17),(16,1))
            array2 = np.reshape(np.arange(17,33),(16,1))
            combined = np.concatenate((array,array2),axis=1).T
            self.upper_grids = combined
        else:
            if self.num_grids == 2:
                array = np.arange(1, 8 * 8 + 1).reshape(8, 8)
                # Rotate the array 90 degrees counterclockwise
                rotated_grid = np.rot90(array, k=1)
                self.upper_grids = np.concatenate(
                    [
                        rotated_grid + (64 * (self.channel_order[0] - 1)),
                        rotated_grid + (64 * (self.channel_order[1] - 1)),
                    ],
                    axis=1,
                )
            elif self.num_grids == 1:
                array = np.arange(1, 8 * 8 + 1).reshape(8, 8)
                # Rotate the array 90 degrees counterclockwise
                rotated_grid = np.rot90(array, k=1)
                self.upper_grids = rotated_grid + (64 * (self.channel_order[0] - 1))
            elif self.num_grids == 3:
                array = np.arange(1, 8 * 8 + 1).reshape(8, 8)
                # Rotate the array 90 degrees counterclockwise
                rotated_grid = np.rot90(array, k=1)
                self.upper_grids = np.concatenate(
                    [
                        rotated_grid + (64 * (self.channel_order[0] - 1)),
                        rotated_grid + (64 * (self.channel_order[1] - 1)),
                        rotated_grid + (64 * (self.channel_order[2] - 1)),
                    ],
                    axis=1,
                )
            elif self.num_grids == 5:
                array = np.arange(1, 8 * 8 + 1).reshape(8, 8)
                # Rotate the array 90 degrees counterclockwise
                rotated_grid = np.rot90(array, k=1)
                self.upper_grids = np.concatenate(
                    [
                        rotated_grid + (64 * (self.channel_order[0] - 1)),
                        rotated_grid + (64 * (self.channel_order[1] - 1)),
                        rotated_grid + (64 * (self.channel_order[2] - 1)),
                    ],
                    axis=1,
                )
                self.lower_grids = np.concatenate(
                    [
                        rotated_grid + (64 * (self.channel_order[3] - 1)),
                        rotated_grid + (64 * (self.channel_order[4] - 1)),
                    ],
                    axis=1,
                )
            else:
                raise ValueError("Number of grids not supported")

    def get_channel_position_and_grid_number(self, number_of_channel):
        """
        returns the row and column of the channel in the grid and the grid number i.e if all grids are normally connected [1,2,3,4,5] then 320 would be in the grid 5, if order is [1,2,3,5,4] then 320 would be in grid 4
        :param number_of_channel: the channel number [1,320] that we want to have the positions in the grid of
        :return: row,col,grid_number
        """
        try:
            row, col = np.where(self.upper_grids == number_of_channel)
            if col[0] == 0:
                grid_number = 1
            else:
                grid_number = np.ceil(col[0] / 8)
            return row[0], col[0], int(grid_number)

        except:
            pass

        try:
            row, col = np.where(self.lower_grids == number_of_channel)
            if col[0] == 0:
                grid_number = 4
            else:
                grid_number = np.ceil(col[0] / 8) + 3
            return row[0], col[0], int(grid_number)
        except:
            pass

    def get_channel_position_and_grid_number_backtrans(self, number_of_channel):
        """
        returns the row and column of the channel in the grid and the grid number i.e if all grids are normally connected [1,2,3,4,5] then 320 would be in the grid 5, if order is [1,2,3,5,4] then 320 would be in grid 4
        :param number_of_channel: the channel number [1,320] that we want to have the positions in the grid of
        :return: row,col,grid_number
        """
        if self.use_muovi_pro:
            grid = self.upper_grids
        elif self.num_grids == 5:
            grid = self.concatenate_upper_and_lower_grid(
                np.reshape(self.upper_grids, (8, 24, 1)),
                np.reshape(self.lower_grids, (8, 16, 1)),
            ).reshape(16, 24)
        else:
            grid = self.upper_grids
        try:
            row, col = np.where(grid == number_of_channel)
            return row[0], col[0]

        except:
            pass

    def transfer_320_into_grid_arangement1(self, input):
        """
        transfers the 320 channels into the grid arrangement either 3 grids or 5 grids for 5 grids we return upper_grid and lower_grid otherwise only upper_grid
        :param input:
        :return: upper_grid,lower_grid (for 5 grids) or upper_grid (for 3 grids)
        """
        if self.use_muovi_pro:
            grid = np.zeros((2,16, min(len(row) for row in input)))
            for row in range(input.shape[0]):
                res_row, res_col = self.get_channel_position_and_grid_number_backtrans(
                    row + 1
                )
                grid[res_row, res_col, :] = input[row]
            return grid, None
        else:
            if self.num_grids <= 3:
                grid = np.zeros((8, 8 * self.num_grids, min(len(row) for row in input)))
                for row in range(input.shape[0]):
                    res_row, res_col, res_grid = self.get_channel_position_and_grid_number(
                        row + 1
                    )
                    grid[res_row, res_col, :] = input[row]
                return grid, None

            elif self.num_grids == 5:
                upper_grid = np.zeros((8, 8 * 3, min(len(row) for row in input)))
                lower_grid = np.zeros((8, 8 * 2, min(len(row) for row in input)))
                for row in range(input.shape[0]):
                    res_row, res_col, res_grid = self.get_channel_position_and_grid_number(
                        row + 1
                    )
                    if res_grid < 4:
                        upper_grid[res_row, res_col, :] = input[
                            row
                        ]
                    else:
                        lower_grid[res_row, res_col, :] = input[
                            row
                        ]
                return upper_grid, lower_grid

    def transfer_320_into_grid_arangement(self, input):
        """
        transfers the 320 channels into the grid arrangement either 3 grids or 5 grids for 5 grids we return upper_grid and lower_grid otherwise only upper_grid
        :param input:
        :return: upper_grid,lower_grid (for 5 grids) or upper_grid (for 3 grids)
        """
        if self.use_muovi_pro:
            grid = np.zeros((2,16, min(len(row) for row in input)))
            for row in range(input.shape[0]):
                res_row, res_col = self.get_channel_position_and_grid_number_backtrans(
                    row + 1
                )
                grid[res_row, res_col, :] = input[row]
            return grid, None
        else:
            if self.num_grids <= 3:
                grid = np.zeros((8, 8 * self.num_grids, min(len(row) for row in input)))
                for row in range(input.shape[0]):
                    res_row, res_col, res_grid = self.get_channel_position_and_grid_number(
                        row + 1
                    )
                    grid[res_row, res_col, :] = input[row]
                return grid, None

            elif self.num_grids == 5:
                upper_grid = np.zeros((8, 8 * 3, min(len(row) for row in input)))
                lower_grid = np.zeros((8, 8 * 2, min(len(row) for row in input)))
                for row in range(input.shape[0]):
                    res_row, res_col, res_grid = self.get_channel_position_and_grid_number(
                        row + 1
                    )
                    if res_grid < 4:
                        upper_grid[res_row, res_col, :] = input[row]
                    else:
                        lower_grid[res_row, res_col, :] = input[row]
                return upper_grid, lower_grid

    #

    def transfer_and_concatenate_320_into_grid_arangement_all_samples(self, input):
        reverse_input = input.transpose()
        reverse_input = self.transfer_and_concatenate_320_into_grid_arangement(reverse_input)
        return reverse_input


    def transfer_and_concatenate_320_into_grid_arangement(self, input):
        """
        transfers the 320 channels into the grid arrangement either 3 grids or 5 grids for 5 grids we return upper_grid and lower_grid otherwise only upper_grid
        :param input:
        :return: upper_grid,lower_grid (for 5 grids) or upper_grid (for 3 grids)
        """
        if self.use_muovi_pro:
            grid = np.zeros((2,16, min(len(row) for row in input)))
            for row in range(input.shape[0]):
                res_row, res_col = self.get_channel_position_and_grid_number_backtrans(
                    row + 1
                )
                grid[res_row, res_col, :] = input[row]
            return grid
        else:
            if self.num_grids <= 3:
                grid = np.zeros((8, 8 * self.num_grids, min(len(row) for row in input)))
                for row in range(input.shape[0]):
                    res_row, res_col, res_grid = self.get_channel_position_and_grid_number(
                        row + 1
                    )
                    grid[res_row, res_col, :] = input[row]
                return grid

            elif self.num_grids == 5:
                upper_grid = np.zeros((8, 8 * 3, min(len(row) for row in input)))
                lower_grid = np.zeros((8, 8 * 2, min(len(row) for row in input)))
                for row in range(input.shape[0]):
                    res_row, res_col, res_grid = self.get_channel_position_and_grid_number(
                        row + 1
                    )
                    if res_grid < 4:
                        upper_grid[res_row, res_col, :] = input[row]
                    else:
                        lower_grid[res_row, res_col, :] = input[row]
                return self.concatenate_upper_and_lower_grid(upper_grid, lower_grid)

    def concatenate_upper_and_lower_grid(self, upper_grid, lower_grid):
        """
        concatenates the upper and lower grid into one array
        :param upper_grid:
        :param lower_grid:
        :return:
        """
        if self.num_grids == 5:
            if np.ndim(upper_grid) == 2:
                lower = np.concatenate(
                    (lower_grid, np.zeros((8, 8))), axis=1
                )
            else:
                lower = np.concatenate(
                    (lower_grid, np.zeros((8, 8, upper_grid.shape[2]))), axis=1
                )
            return np.concatenate((upper_grid, lower), axis=0)
        else:
            raise ValueError("Number of grids not supported")

    def transfer_grid_arangement_into_320(self, input):
        if self.use_muovi_pro:
            extracted_data = np.empty((32, input.shape[2]))
            for channel in range(32):
                res_row, res_col = self.get_channel_position_and_grid_number_backtrans(
                    channel + 1
                )
                extracted_data[channel] = input[res_row, res_col]
            return extracted_data
        else:
            if self.num_grids <= 3:
                extracted_data = np.empty((64 * self.num_grids, input.shape[2]))
                for channel in range(64 * self.num_grids):
                    res_row, res_col = self.get_channel_position_and_grid_number_backtrans(
                        channel + 1
                    )
                    extracted_data[channel] = input[res_row, res_col]
                return extracted_data

            elif self.num_grids == 5:
                extracted_data = np.empty(
                    (320, input.shape[2])
                )
                for channel in range(320):
                    res_row, res_col = self.get_channel_position_and_grid_number_backtrans(
                        channel + 1
                    )
                    extracted_data[channel ] = input[
                        res_row, res_col
                    ]
                return extracted_data
            else:
                raise ValueError("Number of grids not supported")

    def split_grid_into_8x8_grids(self, grid):
        """
        Splits the grid into 8x8 grids.

        :param grid: A numpy array representing the grid i.e with shape f.e.(16, 27)
        :return: A list of 8x8 numpy arrays
        """
        # Initialize an empty list to store the 8x8 grids
        if not self.use_muovi_pro:
            grids = []

            # Iterate over each row in the grid
            for i in range(0, grid.shape[0], 8):
                # Iterate over each column in the grid
                for j in range(0, grid.shape[1], 8):
                    # Extract the 8x8 grid
                    grids.append(grid[i : i + 8, j : j + 8, :])

            return grids
        else:
            print("not supported for muovi pro")
            return None

    def from_grid_position_to_row_position(self, row_position, column_position):
        """
        Converts the grid position to row position.

        :param grid_position: The grid position (row,column)
        :return: The row position (0 - 320)
        """
        if self.use_muovi_pro:
            return self.upper_grids[row_position, column_position]
        elif self.num_grids <= 3:
            return self.upper_grids[row_position, column_position]
        else:
            return self.concatenate_upper_and_lower_grid(self.upper_grids, self.lower_grids)[row_position, column_position]



# a = Grid_Arrangement([2,1])
# a.make_grid()
# con = a.concatenate_upper_and_lower_grid(a.upper_grids,a.lower_grids)
# data =a.transfer_grid_arangement_into_320(np.reshape(a.upper_grids,(8,16,1)) )
# new = a.transfer_320_into_grid_arangement(data)
# electrode_values = np.random.rand(320, 100)
# grid, lower_grid= a.transfer_320_into_grid_arangement(electrode_values)

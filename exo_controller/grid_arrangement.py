import numpy as np

class Grid_Arrangement:
    def __init__(self,channel_order = None):
        self.num_grids = len(channel_order)
        self.channel_order = channel_order # which grid is placed on which quattrocento outlet [2,1,4,5,3] means outlet 2 is the first grid, besides is outlet 1 as second grid...


    def make_grid(self):
        """
        creates the grid arrangement for the quattrocento it creates a self variable upper_grid and lower_grid if 5 grids are used otherwise only upper_grid , upper grid = 3 grids besides , lower grid = 2 besides, furthermore it uses the way how we used the outlets from the quattrocento to fit it to the grid positions
        :return:
        """
        if self.num_grids== 3:
            array = np.arange(1, 8 * 8 + 1).reshape(8, 8)
            # Rotate the array 90 degrees counterclockwise
            rotated_grid = np.rot90(array, k=1)
            self.upper_grids =np.concatenate([rotated_grid + (64 * (self.channel_order[0]-1)), rotated_grid + (64 * (self.channel_order[1]-1)), rotated_grid + (64 * (self.channel_order[2]-1))], axis=1)
        elif self.num_grids == 5:
            array = np.arange(1, 8 * 8 + 1).reshape(8, 8)
            # Rotate the array 90 degrees counterclockwise
            rotated_grid = np.rot90(array, k=1)
            self.upper_grids =np.concatenate([rotated_grid + (64 * (self.channel_order[0]-1)), rotated_grid + (64 * (self.channel_order[1]-1)), rotated_grid + (64 * (self.channel_order[2]-1))], axis=1)
            self.lower_grids=   np.concatenate([rotated_grid+ (64 * (self.channel_order[3]-1)), rotated_grid + (64 * (self.channel_order[4]-1))], axis=1)
        else:
            raise ValueError("Number of grids not supported")

    def get_channel_position_and_grid_number(self,number_of_channel):
        """
        returns the row and column of the channel in the grid and the grid number i.e if all grids are normally connected [1,2,3,4,5] then 320 would be in the grid 5, if order is [1,2,3,5,4] then 320 would be in grid 4
        :param number_of_channel: the channel number [1,320] that we want to have the positions in the grid of
        :return: row,col,grid_number
        """
        try:
            row, col = np.where(self.upper_grids == number_of_channel)
            grid_number = np.ceil(col / 8)
            return row[0], col[0], int(grid_number[0])

        except:
            pass

        try:
            row, col = np.where(self.lower_grids == number_of_channel)
            grid_number = np.ceil(col / 8) + 3
            return row[0], col[0], int(grid_number[0])
        except:
            pass



    def transfer_320_into_grid_arangement(self,input):
        """
        transfers the 320 channels into the grid arrangement either 3 grids or 5 grids for 5 grids we return upper_grid and lower_grid otherwise only upper_grid
        :param input:
        :return: upper_grid,lower_grid (for 5 grids) or upper_grid (for 3 grids)
        """
        if self.num_grids == 3:
            grid = np.empty((8, 8 * 3, min(len(row) for row in input)))
            for row in range (input.shape[0]):
                res_row,res_col,res_grid = self.get_channel_position_and_grid_number(row+1)
                grid[res_row,res_col,:] = input[row]
            return grid,None

        elif self.num_grids == 5:
            upper_grid = np.empty((8, 8 * 3, min(len(row) for row in input)))
            lower_grid = np.empty((8, 8 * 2, min(len(row) for row in input)))
            for row in range (input.shape[0]):
                res_row,res_col,res_grid = self.get_channel_position_and_grid_number(row+1)
                if res_grid < 4:
                    upper_grid[res_row,res_col,:] = input[row]
                else:
                    lower_grid[res_row,res_col,:] = input[row]
            return upper_grid,lower_grid

    def concatenate_upper_and_lower_grid(self,upper_grid,lower_grid):
        """
        concatenates the upper and lower grid into one array
        :param upper_grid:
        :param lower_grid:
        :return:
        """
        if self.num_grids == 5:
            lower  = np.concatenate((lower_grid,np.zeros((8,8,upper_grid.shape[2]))),axis=1)
            return np.concatenate((upper_grid,lower),axis=0)
        else:
            raise ValueError("Number of grids not supported")

    def transfer_grid_arangement_into_320(self,input):
        if self.num_grids == 3:
            extracted_data = np.empty((192, input[5,5].shape[0]))
        elif self.num_grids == 5:
            extracted_data = np.empty((320, input.shape[2]))
        else:
            raise ValueError("Number of grids not supported")

        # TODO implement this




a = Grid_Arrangement([2,1,3,5,4])
a.make_grid()
electrode_values = np.random.rand(320, 100)
grid, lower_grid= a.transfer_320_into_grid_arangement(electrode_values)
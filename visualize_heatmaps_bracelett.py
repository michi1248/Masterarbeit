import matplotlib.pyplot as plt

from exo_controller.helpers import *
from exo_controller import grid_arrangement
from exo_controller import normalizations
from exo_controller.helpers import *
import seaborn as sns
from PIL import Image
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from exo_controller.helpers import *
from scipy.signal import resample
import gc

from exo_controller import *


class Heatmap:
    def __init__(
        self,
        movement_name,
        path_to_subject_dat,
        path_to_data,
        sampling_frequency=2048,
        path_to_save_plots=r"D:\Lab\differences_train_test_heatmaps",
        frame_duration=150,
        additional_term="",
        method="roboust",
        mean_flex_rest=None,
        mean_ex_rest=None,
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

        self.normalizer = normalizations.Normalization(
            method=method,
            grid_order=[1, 2, 3, 4, 5],
            frame_duration=frame_duration,
            use_muovi_pro=True,
            skip_in_samples=30,
        )

        self.grid_aranger = grid_arrangement.Grid_Arrangement([1, 2, 3, 4, 5],use_muovi_pro=True)
        self.grid_aranger.make_grid()

        if (mean_flex_rest is not None) and (mean_ex_rest is not None):
            self.normalizer.set_mean(mean=mean_ex_rest)
        self.get_all_emg_data(
            path_to_data=path_to_subject_dat,
        )
        self.normalizer.calculate_normalization_values()
        # self.max_for_heatmap,self.min_for_heatmap = self.normalizer.calculate_norm_values_heatmap()
        # self.max_for_heatmap = self.normalizer.normalize_chunk(self.max_for_heatmap)
        # self.min_for_heatmap = self.normalizer.normalize_chunk(self.min_for_heatmap)

        self.gauss_filter = create_gaussian_filter(size_filter=3)

        self.movement_name = movement_name
        self.mean_ex = mean_ex_rest
        self.mean_flex = mean_flex_rest

        if not os.path.exists(
            os.path.join(
                path_to_save_plots,
                str(frame_duration) + "ms_rms_window",
                method,
                movement_name + "_" + additional_term,
            )
        ):
            os.makedirs(
                os.path.join(
                    path_to_save_plots,
                    str(frame_duration) + "ms_rms_window",
                    method,
                    movement_name + "_" + additional_term,
                )
            )
        self.path_to_save_plots = os.path.join(
            path_to_save_plots,
            str(frame_duration) + "ms_rms_window",
            method,
            movement_name + "_" + additional_term,
        )
        self.emg_data = (
            load_pickle_file(path_to_data)[
                "emg"
            ]
            # .transpose(1, 0, 2)
            # .reshape(64*3, -1)
        )



        # emg_data_for_max_min = load_pickle_file(os.path.join(path_to_subject_dat, "emg_data.pkl"))
        # for i in emg_data_for_max_min.keys():
        #     emg_data_for_max_min[i] = np.array(emg_data_for_max_min[i].transpose(1,0,2).reshape(320,-1))
        #     if (self.mean_ex is not None) and (self.mean_flex is not None):
        #         # have to transfer self.mean_ex from grid arrangement to 320 channels arangement
        #         emg_data_for_max_min[i] = emg_data_for_max_min[i] - grid_aranger.transfer_grid_arangement_into_320(np.reshape(self.mean_ex,(self.mean_ex.shape[0],self.mean_ex.shape[1],1)))

        if method == "EMG_signals":
            plot_emg_channels(
                self.emg_data,
                save_path=self.path_to_save_plots,
                movement=self.movement_name + "_raw",
            )

        #
        # if method == "Min_Max_Scaling_all_channels":
        #     self.max_values, self.min_values = find_max_min_values_for_each_movement_(emg_data_for_max_min, range(320),
        #                                                                               list(emg_data_for_max_min.keys()))
        #
        # if method == "Min_Max_Scaling_over_whole_data" :
        #     self.max_values,self.min_values = find_max_min_values_for_each_movement_and_channel(emg_data_for_max_min,range(320),list(emg_data_for_max_min.keys()))
        #     self.max_values = self.max_values.reshape(320,1)
        #     self.min_values = self.min_values.reshape(320,1)
        #
        # if (method == "Robust_Scaling") or (method == "Gauss_filter"):
        #     self.q1,self.q2,self.median = find_q_median_values_for_each_movement_and_channel(emg_data_for_max_min,range(320),list(emg_data_for_max_min.keys()))
        #     self.q1 = self.q1.reshape(320,1)
        #     self.q2 = self.q2.reshape(320,1)
        #     self.median = self.median.reshape(320,1)
        #
        # if method == "Robust_all_channels":
        #     self.q1, self.q2, self.median = find_q_median_values_for_each_movement(emg_data_for_max_min, range(320), list(
        #         emg_data_for_max_min.keys()))

        self.emg_data = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement(
            self.emg_data
        )
        # if method == "Min_Max_Scaling_over_whole_data":
        #     upper_max,lower_max = grid_aranger.transfer_320_into_grid_arangement(self.max_values)
        #     upper_min,lower_min = grid_aranger.transfer_320_into_grid_arangement(self.min_values)
        #
        # if (method == "Robust_Scaling") or (method == "Gauss_filter"):
        #     upper_q1,lower_q1 = grid_aranger.transfer_320_into_grid_arangement(self.q1)
        #     upper_q2,lower_q2 = grid_aranger.transfer_320_into_grid_arangement(self.q2)
        #     upper_median, lower_median = grid_aranger.transfer_320_into_grid_arangement(self.median)


        self.ref_data = load_pickle_file(
            path_to_data
        )["kinematics"]

        # resample ref to same length like emg
        new_ref_data = np.zeros((self.ref_data.shape[0], self.emg_data.shape[2]))
        for i in range (self.ref_data.shape[0]):
            new_ref_data[i,:] =  resample(
                self.ref_data[i,:], self.emg_data.shape[2]
            )
        self.ref_data = new_ref_data

        print("ref_data shape: ", self.ref_data.shape)
        print("emg_data shape: ", self.emg_data.shape)

        # if method == "Min_Max_Scaling_over_whole_data":
        #     self.max_values = grid_aranger.concatenate_upper_and_lower_grid(upper_max,lower_max)
        #     self.min_values = grid_aranger.concatenate_upper_and_lower_grid(upper_min, lower_min)
        # if (method == "Robust_Scaling") or (method == "Gauss_filter"):
        #     self.median = grid_aranger.concatenate_upper_and_lower_grid(upper_median, lower_median)
        #     self.q1 = grid_aranger.concatenate_upper_and_lower_grid(upper_q1, lower_q1)
        #     self.q2 = grid_aranger.concatenate_upper_and_lower_grid(upper_q2, lower_q2)
        self.sampling_frequency = sampling_frequency

        self.sample_length = self.emg_data[1][1].shape[0]
        self.num_samples = int(
            self.sample_length / (self.sampling_frequency * (frame_duration / 1000))
        )
        self.frame_duration = frame_duration
        self.number_observation_samples = int(
            (self.frame_duration / 1000) * self.sampling_frequency
        )

        self.fig, (self.ax_emg, self.ax_ref) = plt.subplots(2, 1, figsize=(8, 10))
        self.ax_ref.set_title("Reference")
        self.ax_ref.set(
            xlabel="Time (s)",
            ylabel="Amplitude (mV)",
            xlim=(0, self.sample_length / self.sampling_frequency),
            ylim=(0, 1),
        )
        self.x_for_ref = np.linspace(
            0, self.sample_length / self.sampling_frequency, self.ref_data.shape[1]
        )
        self.global_counter = 0
        self.last_frame = 0

    def get_all_emg_data(self, path_to_data):

        all_emg_data = []

        for recording in os.listdir(path_to_data):
            a = load_pickle_file(os.path.join(path_to_data, recording))["emg"]

            emg_data_one_movement = (
                self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement(
                    a
                )
            )
            all_emg_data.append(emg_data_one_movement)
        self.normalizer.all_emg_data = all_emg_data

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
        if self.movement_name == "rest":
            self.heatmaps_flex = np.add(self.heatmaps_flex, heatmap)
            # add heatmap and not normalized heatmap because impact of all heatmaps will be the same but maybe some heatmaps have higher values and i want to use this ??
            # TODO maybe change this (better to use normalized or not ?)
            # IT IS BETER TO USE NORMALIZED BECAUSE IF EMG SCHWANKUNGEN
            self.number_heatmaps_flex += 1
            self.heatmaps_ex = np.add(self.heatmaps_ex, heatmap)
            self.number_heatmaps_ex += 1

        if (self.mean_ex is not None) and (self.mean_flex is not None):
            heatmap = heatmap - self.mean_ex

        normalized_heatmap = self.normalizer.normalize_chunk(heatmap)

        normalized_heatmap = apply_gaussian_filter(
            normalized_heatmap, self.gauss_filter
        )



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
                    self.heatmaps_flex, normalized_heatmap
                )  # np.multiply(heatmap, 1/(distance+0.1) ))
                # add heatmap and not normalized heatmap because impact of all heatmaps will be the same but maybe some heatmaps have higher values and i want to use this ??
                # TODO maybe change this (better to use normalized or not ?)
                # IT IS BETER TO USE NORMALIZED BECAUSE IF EMG SCHWANKUNGEN
                self.number_heatmaps_flex += 1
                # plot_local_maxima_minima(
                #     self.ref_data[:, 0],
                #     self.local_maxima,
                #     self.local_minima,
                #     current_sample_position=frame,
                #     color="black",
                # )
            else:
                self.heatmaps_ex = np.add(
                    self.heatmaps_ex, normalized_heatmap
                )  # np.multiply(heatmap, 1/(distance+0.1) ))
                self.number_heatmaps_ex += 1
                # plot_local_maxima_minima(
                #     self.ref_data[:, 0],
                #     self.local_maxima,
                #     self.local_minima,
                #     current_sample_position=frame,
                #     color="yellow",
                # )

        if self.global_counter == 0:
            hmap = sns.heatmap(
                normalized_heatmap,
                ax=self.ax_emg,
                cmap="RdBu_r",
                cbar=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={"label": "norm. RMS"},
            )  # ,cbar_ax=self.ax_emg,)

        else:
            hmap = sns.heatmap(
                normalized_heatmap,
                ax=self.ax_emg,
                cmap="RdBu_r",
                cbar=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={"label": "norm. RMS"},
            )  # ,cbar_ax=self.ax_emg,)

        self.global_counter += 1
        # self.emg_plot.set(normalized_heatmap, ax=self.ax_emg, cmap='hot', cbar=True, xticklabels=True, yticklabels=True, cbar_kws={'label': 'norm. RMS'},mask=mask_for_heatmap)
        # self.ax_emg.imshow(heatmap, cmap='hot', interpolation='nearest')

    def make_Ref_trackings(self, frame):

        index_movement = 0
        max_difference = 0
        for i in range(self.ref_data.shape[0]):
            if np.max(self.ref_data[i, :]) - np.min(self.ref_data[i, 0]) > max_difference:
                max_difference = np.max(self.ref_data[:, i]) - np.min(self.ref_data[:, i])
                index_movement = i

        self.ax_ref.plot(
            self.x_for_ref,
            self.ref_data[index_movement,:],
            color="blue",
        )
        self.ax_ref.scatter(
            self.x_for_ref[frame],
            self.ref_data[index_movement][frame],
            color="green",
            marker="x",
            s=90,
            linewidth=3,
        )

    def update(self, frame):
        plt.figure()
        self.fig, (self.ax_emg, self.ax_ref) = plt.subplots(2, 1, figsize=(8, 10))
        self.ax_ref.set_title("Reference")
        self.ax_ref.set(
            xlabel="Time (s)",
            ylabel="Amplitude (mV)",
            xlim=(0, self.sample_length / self.sampling_frequency),
            ylim=(0, 1),
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
        self.make_Ref_trackings(frame)
        # Save the frame as an image
        frame_filename = os.path.join(self.path_to_save_plots, f"frame_{frame:03d}.png")
        self.fig.canvas.draw()
        # plt.savefig(frame_filename)
        pil_image = Image.frombytes(
            "RGB", self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb()
        )
        pil_image.save(frame_filename)
        self.pbar.update(1)
        plt.close()

        return self.ax_emg, self.ax_ref

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
        self.samples = [i for i in range(0, self.ref_data.shape[1], 64)]
        self.samples = [
            element for element in self.samples if element <= self.ref_data.shape[1]
        ]
        self.samples = self.samples[:int(0.3*len(self.samples))]
        # make both lists to save all coming heatmaps into it by adding the values and dividing at the end through number of heatmaps
        num_rows, num_cols, _ = self.emg_data.shape
        self.heatmaps_flex = np.zeros((num_rows, num_cols))
        self.heatmaps_ex = np.zeros((num_rows, num_cols))
        self.number_heatmaps_flex = 0
        self.number_heatmaps_ex = 0
        index_movement = 0
        max_difference = 0
        for i in range(self.ref_data.shape[0]):
            if np.max(self.ref_data[i, :]) - np.min(self.ref_data[i, 0]) > max_difference:
                max_difference = np.max(self.ref_data[:, i]) - np.min(self.ref_data[:, i])
                index_movement = i


        self.local_maxima, self.local_minima = get_locations_of_all_maxima(
            self.ref_data[index_movement, :]
        )
        plot_local_maxima_minima(
            self.ref_data[index_movement, :], self.local_maxima, self.local_minima
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
        # plt.show()

    def channel_extraction(self, mark_choosen_channels=False):
        self.pbar.close()
        ########

        mean_flex_heatmap = np.divide(self.heatmaps_flex, self.number_heatmaps_flex)

        mean_ex_heatmap = np.divide(self.heatmaps_ex, self.number_heatmaps_ex)


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

        fig, axes = plt.subplots(3, 1, figsize=(6, 8))
        # difference heatmap
        axes[0].set_title("Difference Heatmap")
        sns.heatmap(difference_heatmap, ax=axes[0], cmap="RdBu_r")
        if mark_choosen_channels:
            axes[1].set_title("chosen flexion channels")
            sns.heatmap(chosen_heatmap_flexion, ax=axes[1], cmap="RdBu_r")
            axes[2].set_title("chosen extension channels")
            sns.heatmap(chosen_heatmap_extension, ax=axes[2], cmap="RdBu_r")
            plt.savefig(
                os.path.join(
                    path_to_difference_plot,
                    "difference_plot_chosen_" + str(movement) + ".png",
                )
            )
        else:
            axes[1].set_title("Mean Flexion Heatmap")
            sns.heatmap(mean_flex_heatmap, ax=axes[1], cmap="RdBu_r")
            axes[2].set_title("Mean Extension Heatmap")
            sns.heatmap(mean_ex_heatmap, ax=axes[2], cmap="RdBu_r")
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

    # "Min_Max_Scaling_over_whole_data" = min max scaling with max/min is choosen individually for every channel
    # "Robust_all_channels" = robust scaling with q1,q2,median is choosen over all channels
    # "Robust_Scaling"  = robust scaling with q1,q2,median is choosen individually for every channel
    # "Min_Max_Scaling_all_channels" = min max scaling with max/min is choosen over all channels
    # "EMG_signals" = plot emg signals
    # "Gauss_filter" = use Gauss filter on raw data
    # "Bandpass_filter" = use bandpass filter on raw data and plot the filtered all emg channels
    plot_emg = False

    window_size = 150
    for method in [
        #"Min_Max_Scaling_over_whole_data",
        #"Robust_Scaling",
        "Min_Max_Scaling_all_channels",
        #"no_scaling",
        #"Robust_all_channels",
        # "EMG_signals",
    ]:
        mean_flex_rest = None
        mean_ex_rest = None

        for additional_term in [""]:#["before", "after"]:

            # if additional_term == "before":
            #     path = r"D:\Lab\MasterArbeit\trainings_data\resulting_trainings_data\subject_Michi_18_01_2024_normal2"  # trainingsdata recorded for training
            # else:
            #     path = r"D:\Lab\MasterArbeit\trainings_data\resulting_trainings_data\subject_Michi_18_01_2024_normal3"  # trainingsdata recorded after training
            path = r"D:\Lab\SCI_recording_Brandmueller_23_2\data\recordings\recording1"
            for movement in ["rest", "fist", "middle", "pinky", "index", "thumb", "ring"]:
                for i, s in enumerate(os.listdir(path)):
                    if movement in s:
                        index = i
                recording = os.listdir(path)[index]

                if movement in recording:
                    path1 = os.path.join(path, recording)

                    print("method is: ", method)
                    print("movement is: ", movement)
                    print("additional term is: ", additional_term)

                    heatmap = Heatmap(
                        movement_name=movement,
                        path_to_data=path1,
                        path_to_subject_dat=path,
                        frame_duration=window_size,
                        additional_term=additional_term,
                        method=method,
                    )
                    if method == "EMG_signals":
                        continue
                    heatmap.animate(save=True)
                    fps = 10
                    heatmap.save_animation(
                        r"D:\Lab\differences_train_test_heatmaps/"
                        + str(window_size)
                        + "ms_rms_window/"
                        + method
                        + "/"
                        + str(movement)
                        + str(fps)
                        + "fps_most_firings.gif",
                        fps=fps,
                    )
                    mean_flex, mean_ex = heatmap.channel_extraction()
                    heatmap.channel_extraction(mark_choosen_channels=True)

                    if plot_emg:
                        # Do the same again for method == EMG_signals to produce emg singals plot
                        heatmap = Heatmap(
                            movement,
                            path,
                            frame_duration=window_size,
                            additional_term=additional_term,
                            method="EMG_signals",
                        )

                    if movement == "rest":
                        # if the movement is rest save the mean flex and ex for later subtraction from the other movements
                        mean_ex_rest = mean_ex
                        mean_flex_rest = mean_flex

                    else:
                        # if the movement is not rest subtract the mean flex and ex from the rest from the current movement therefore give the means of the rest to the functions
                        heatmap = Heatmap(
                            movement_name=movement,
                            path_to_data=path1,
                            path_to_subject_dat=path,
                            frame_duration=window_size,
                            additional_term=additional_term
                            + "_subtracted_mean_rest_from_emg",
                            method=method,
                            mean_flex_rest=mean_flex_rest,
                            mean_ex_rest=mean_ex_rest,
                        )
                        heatmap.animate(save=True)
                        heatmap.save_animation(
                            r"D:\Lab\differences_train_test_heatmaps/"
                            + str(window_size)
                            + "ms_rms_window/"
                            + method
                            + "/"
                            + str(movement)
                            + str(fps)
                            + "fps_most_firings_trash.gif",
                            fps=10,
                        )
                        mean_flex, mean_ex = heatmap.channel_extraction()
                        heatmap.channel_extraction(mark_choosen_channels=True)

                        if plot_emg:
                            # Do the same again for method == EMG_signals to produce emg singals plot
                            heatmap = Heatmap(
                                movement,
                                path,
                                frame_duration=window_size,
                                additional_term=additional_term
                                + "_subtracted_mean_rest_from_emg",
                                method="EMG_signals",
                                mean_flex_rest=mean_flex_rest,
                                mean_ex_rest=mean_ex_rest,
                            )
                    gc.collect()
    # movement = "thumb"
    # heatmap = Heatmap(movement,r"D:\Lab\MasterArbeit\trainings_data\resulting_trainings_data\subject_Michi_Test1",additional_term="before")
    # heatmap.animate(save=True)
    #
    # fps = 2
    # heatmap.save_animation(r"D:\Lab\differences_train_test_heatmaps/" +str(movement) +str(fps) +"fps_most_firings.gif", fps=2)
    # heatmap.channel_extraction()
    # heatmap.channel_extraction(mark_choosen_channels=True)

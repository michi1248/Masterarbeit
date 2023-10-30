from exo_controller.helpers import *
from exo_controller import grid_arrangement

import seaborn as sns
from PIL import Image
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from exo_controller.helpers import *
from scipy.signal import resample

from exo_controller import *





class Heatmap:
    def __init__(self, movement_name,path_to_subject_dat,sampling_frequency=2048,path_to_save_plots=r"D:\Lab\differences_train_test_heatmaps",frame_duration=150,additional_term = ""):
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
        print("NOTE !!!! at the moment if there are more than one ref dimension is given (2pinch,fist,etc,) for the plot the first one is taken!!!",file=sys.stderr )

        self.movement_name = movement_name
        if not os.path.exists(os.path.join(path_to_save_plots, movement_name+ "_" + additional_term)):
            os.makedirs(os.path.join(path_to_save_plots, movement_name+ "_" + additional_term))
        self.path_to_save_plots = os.path.join(path_to_save_plots, movement_name+ "_" + additional_term)
        self.emg_data = load_pickle_file(os.path.join(path_to_subject_dat, "emg_data.pkl"))[movement_name].transpose(1,0,2).reshape(320,-1)
        grid_aranger = grid_arrangement.Grid_Arrangement([1,2,3,4,5])
        grid_aranger.make_grid()
        upper,lower = grid_aranger.transfer_320_into_grid_arangement(self.emg_data)
        self.ref_data= load_pickle_file(os.path.join(path_to_subject_dat, "3d_data.pkl"))[movement_name]
        resampling_factor = 2048 / 120
        num_samples_emg = self.emg_data.shape[1]
        num_samples_resampled_ref = int(self.ref_data.shape[0] * resampling_factor)
        resampled_ref = np.empty((num_samples_resampled_ref,2))

        self.ref_data = resample(self.ref_data, num_samples_resampled_ref)


        self.emg_data= grid_aranger.concatenate_upper_and_lower_grid(upper,lower)
        self.sampling_frequency = sampling_frequency
        self.fig, (self.ax_emg, self.ax_ref) = plt.subplots(2, 1, figsize=(8, 10))
        self.sample_length = self.emg_data[1][1].shape[0]
        self.num_samples = int(self.sample_length/(self.sampling_frequency*(frame_duration/1000)))
        self.frame_duration = frame_duration
        self.number_observation_samples = int((self.frame_duration / 1000) * self.sampling_frequency)
        self.ax_ref.set_title("Reference")
        self.ax_ref.set(xlabel="Time (s)", ylabel="Amplitude (mV)",xlim=(0, self.sample_length / self.sampling_frequency), ylim=(0, 1))
        self.fig.suptitle('Heatmmap comparison of ' + movement_name, fontsize=14)
        self.x_for_ref = np.linspace(0, self.sample_length / self.sampling_frequency, self.ref_data.shape[0])
        self.global_counter=0
        self.last_frame = 0

    def calculate_heatmap(self,emg_grid, position, interval_in_samples):
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
                if (position - interval_in_samples < 0) or (len(emg_grid[row_idx][col_idx]) < (interval_in_samples)):
                    channel_data = emg_grid[row_idx][col_idx][:position + 1]
                else:
                    channel_data = emg_grid[row_idx][col_idx][position - interval_in_samples:position]
                # print(np.sqrt(np.mean(np.array(channel_data) ** 2)))
                rms_values[row_idx, col_idx] = np.sqrt(np.mean(np.array(channel_data) ** 2))

        return rms_values

    def make_heatmap_emg(self,frame):
        """
        :return: heatmap of emg data

        """
        heatmap = self.calculate_heatmap(self.emg_data, frame, self.number_observation_samples)
        normalized_heatmap = normalize_2D_array(heatmap)

        #only do the following if +- window size near extrema
        if (is_near_extremum(frame,self.local_maxima,self.local_minima,time_window=self.frame_duration, sampling_frequency=self.sampling_frequency)):
            # find out to which part of the movement the current sample belongs (half of flex or ref)
            belongs_to_movement,distance = check_to_which_movement_cycle_sample_belongs(frame,self.local_maxima,self.local_minima)
            # add the heatmap to the list of all heatmaps of the fitting flex/ex for later calculation of difference heatmap
            if (belongs_to_movement == 1):
                # the closer the sample is to the extrema the more impact it has on the heatmap
                self.heatmaps_flex = np.add(self.heatmaps_flex, np.multiply(normalized_heatmap, 1/(distance+0.1) ))
                # add heatmap and not normalized heatmap because impact of all heatmaps will be the same but maybe some heatmaps have higher values and i want to use this ??
                #TODO maybe change this (better to use normalized or not ?)
                #IT IS BETER TO USE NORMALIZED BECAUSE IF EMG SCHWANKUNGEN
                self.number_heatmaps_flex +=1
            else:
                self.heatmaps_ex = np.add(self.heatmaps_ex,  np.multiply(normalized_heatmap, 1/(distance+0.1) ))
                self.number_heatmaps_ex += 1

        if self.global_counter == 0:
            hmap = sns.heatmap(normalized_heatmap, ax=self.ax_emg, cmap='RdBu_r', cbar=True, xticklabels=True, yticklabels=True, cbar_kws={'label': 'norm. RMS'})#,cbar_ax=self.ax_emg,)
        else:
            hmap = sns.heatmap(normalized_heatmap, ax=self.ax_emg, cmap='RdBu_r', cbar=False, xticklabels=True, yticklabels=True, cbar_kws={'label': 'norm. RMS'})#,cbar_ax=self.ax_emg,)
        self.global_counter += 1
        #self.emg_plot.set(normalized_heatmap, ax=self.ax_emg, cmap='hot', cbar=True, xticklabels=True, yticklabels=True, cbar_kws={'label': 'norm. RMS'},mask=mask_for_heatmap)
        #self.ax_emg.imshow(heatmap, cmap='hot', interpolation='nearest')

    def make_Ref_trackings(self,frame):
        if ("pinch" in self.movement_name) or ("fist" in self.movement_name):
            self.ax_ref.plot(self.x_for_ref, normalize_2D_array(self.ref_data[:,0]), color='blue')
            self.ax_ref.scatter(self.x_for_ref[frame], normalize_2D_array(self.ref_data[:,0])[frame], color='green',
                                marker='x', s=90, linewidth=3)
        else:
            self.ax_ref.plot(self.x_for_ref, normalize_2D_array(self.ref_data), color='blue')
            self.ax_ref.scatter(self.x_for_ref[frame], normalize_2D_array(self.ref_data)[frame], color='green', marker='x', s=90, linewidth=3)


    def update(self, frame):

        self.ax_emg.clear()
        self.ax_ref.clear()

        if len(self.fig.axes) > 4:
            self.fig.delaxes(self.fig.axes[4])
            self.ax_emg.relim()
            self.ax_emg.autoscale_view()

        self.fig.suptitle('Heatmmap comparison of ' + self.movement_name + "at sample: " + str(frame) + " with window size " + str(self.frame_duration) + " ms", fontsize=12)
        self.make_heatmap_emg(frame)
        self.make_Ref_trackings(frame)
        # Save the frame as an image
        frame_filename = os.path.join(self.path_to_save_plots, f'frame_{frame:03d}.png')
        self.fig.canvas.draw()
        #plt.savefig(frame_filename, bbox_inches='tight', dpi=100)
        pil_image = Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb())
        pil_image.save(frame_filename)
        self.pbar.update(1)

        return self.ax_emg,self.ax_ref

    def update_slider(self,event):
        print(self.slider.get())
        self.update(int(self.slider.get()))

    def close_plot(self):
        self.root.quit()
        self.root.destroy()

    def move_to_closest(self):
        current_value =int(self.slider.get())
        print("current value is: " + str(current_value))
        closest_value = min(self.samples, key=lambda x: abs(x - current_value))
        index_of_closest_value = self.samples.index(closest_value)
        next_higher_value = self.samples[index_of_closest_value + 1]
        print("next value is: " + str(next_higher_value))
        self.slider.set(next_higher_value)

    def animate(self,save=False):
        #samples = all sample values when using all samples with self.frame_duration in between
        self.samples = np.linspace(0, self.sample_length, self.num_samples, endpoint=False, dtype=int)
        self.samples = [element for element in self.samples if element <= self.ref_data.shape[0]]
        #make both lists to save all coming heatmaps into it by adding the values and dividing at the end through number of heatmaps
        num_rows, num_cols,_ = self.emg_data.shape
        self.heatmaps_flex  = np.zeros((num_rows, num_cols))
        self.heatmaps_ex = np.zeros((num_rows, num_cols))
        self.number_heatmaps_flex = 0
        self.number_heatmaps_ex = 0
        if ("pinch" in self.movement_name) or ("fist" in self.movement_name) or ("rest" in self.movement_name):
            self.local_maxima, self.local_minima = get_locations_of_all_maxima(self.ref_data[:,0])
        else:
            self.local_maxima,self.local_minima = get_locations_of_all_maxima(self.ref_data[:])



        if not save:
            #stuff for interactive plot
            self.root = tk.Tk()
            self.root.title("Interactive Heatmap Plot")
            self.label = ttk.Label(self.root, text="Sample")
            self.label.pack()
            self.slider = tk.Scale(self.root, from_=0, to=self.sample_length,orient=tk.HORIZONTAL,length=600,command=self.update_slider)
            self.slider.set(self.samples[0])
            self.slider.pack()
            # Create a "Next" button to trigger the move_to_closest function
            next_button = tk.Button(self.root, text="Next", command=self.move_to_closest)
            next_button.pack()
            #self.slider = widgets.SelectionSlider(options=samples, value=0, description='Sample', continuous_update=False)
            #self.interactive_plot = widgets.interact(self.update, frame=slider)
            #self.slider.bind("<ButtonRelease-1>", self.update_slider)
            #self.slider.bind("<ScaleChanged>", self.update_slider)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
            self.canvas.get_tk_widget().pack()
            # Create a button to close the plot window
            close_button = ttk.Button(self.root, text="Close Plot", command=self.close_plot)
            close_button.pack()
            # Start the Tkinter main loop
            self.root.mainloop()

        if save:
            # for tqdm
            num_frames = len(self.samples)
            self.pbar = tqdm.tqdm(total=num_frames, position=0, leave=True)
            self.ani = FuncAnimation(self.fig, self.update, frames=self.samples[:],cache_frame_data=False )
            #self.ani.event_source.stop()
        plt.show()

    def channel_extraction(self,mark_choosen_channels=False):

        self.pbar.close()
        ########
        mean_flex_heatmap = normalize_2D_array(np.divide(self.heatmaps_flex, self.number_heatmaps_flex))
        mean_ex_heatmap = normalize_2D_array(np.divide(self.heatmaps_ex, self.number_heatmaps_ex))
        difference_heatmap = normalize_2D_array(np.abs(np.subtract(mean_ex_heatmap, mean_flex_heatmap)))
        if(mark_choosen_channels):
            channels_flexion,channels_extension = choose_possible_channels(difference_heatmap,mean_flex_heatmap,mean_ex_heatmap)

            print("choosen channels flexion: ",channels_flexion)
            print("choosen channels extension: ", channels_extension)
            chosen_heatmap_flexion = np.zeros((mean_flex_heatmap.shape[0], mean_flex_heatmap.shape[1]))
            for i in channels_flexion:
                chosen_heatmap_flexion[i[0]][i[1]] = 1
            chosen_heatmap_extension = np.zeros((mean_ex_heatmap.shape[0], mean_ex_heatmap.shape[1]))
            for i in channels_extension:
                chosen_heatmap_extension[i[0]][i[1]] = 1
        path_to_difference_plot = os.path.join(self.path_to_save_plots, "difference_plots")
        if not os.path.exists(path_to_difference_plot):
            os.makedirs(path_to_difference_plot)


        fig, axes = plt.subplots(3, 1, figsize=(6, 8))
        # difference heatmap
        axes[0].set_title("Difference Heatmap")
        sns.heatmap(difference_heatmap, ax=axes[0], cmap='RdBu_r')
        if (mark_choosen_channels):
            axes[1].set_title("chosen flexion channels")
            sns.heatmap(chosen_heatmap_flexion, ax=axes[1], cmap='RdBu_r')
            axes[2].set_title("chosen extension channels")
            sns.heatmap(chosen_heatmap_extension, ax=axes[2], cmap='RdBu_r')
            plt.savefig(os.path.join(path_to_difference_plot, "difference_plot_chosen_" + str(movement) + ".png"))
        else:
            axes[1].set_title("Mean Flexion Heatmap")
            sns.heatmap(mean_flex_heatmap, ax=axes[1], cmap='RdBu_r')
            axes[2].set_title("Mean Extension Heatmap")
            sns.heatmap(mean_ex_heatmap, ax=axes[2], cmap='RdBu_r')
            plt.savefig(os.path.join(path_to_difference_plot, "difference_plot_" + str(movement) + ".png"))
        # chosen channels flexion heatmap
        # sns.heatmap(chosen_heatmap_flexion,ax=axes[1])
        # plt.subplot(3, 1)
        # # chosen channels flexion heatmap
        # sns.heatmap(chosen_heatmap_extension, ax=axes[2])
        #plt.savefig(os.path.join(path_to_difference_plot, "difference_plot" + str(movement) + ".png"))
        ####

    def save_animation(self, output_filename, fps=10):
        self.ani.save(output_filename, writer='pillow', fps=fps)

if __name__ == "__main__":

    movement = "thumb"
    heatmap = Heatmap(movement,r"D:\Lab\MasterArbeit\trainings_data\resulting_trainings_data\subject_Michi_Test1",additional_term="before")
    heatmap.animate(save=True)

    fps = 2
    heatmap.save_animation(r"D:\Lab\differences_train_test_heatmaps/" +str(movement) +str(fps) +"fps_most_firings.gif", fps=2)
    heatmap.channel_extraction()
    heatmap.channel_extraction(mark_choosen_channels=True)

import seaborn as sns
from PIL import Image
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from exo_controller.helpers import *
from exo_controller import *


class Heatmap:
    def __init__(
        self,
        movement_name,
        path_to_subject_dat,
        sampling_frequency=2048,
        path_to_save_plots=r"D:\Lab\MasterArbeit\plots\Heatmaps",
        frame_duration=150,
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

        self.movement_name = movement_name
        if not os.path.exists(os.path.join(path_to_save_plots, movement_name)):
            os.makedirs(os.path.join(path_to_save_plots, movement_name))
        self.path_to_save_plots = os.path.join(path_to_save_plots, movement_name)
        (
            self.emg_data,
            self.Mu_data,
            self.ref_data,
        ) = open_all_files_for_one_patient_and_movement(
            path_to_subject_dat, movement_name
        )
        self.emg_data, self.mask_for_heatmap = fill_empty_array_indexes_with_0(
            self.emg_data
        )
        self.ref_data = self.ref_data.to_numpy()
        self.sampling_frequency = sampling_frequency
        self.fig, (self.ax_emg, self.ax_MU, self.ax_ref) = plt.subplots(
            3, 1, figsize=(8, 10)
        )
        self.sample_length = self.emg_data[0][1][0].shape[0]
        self.num_samples = int(
            self.sample_length / (self.sampling_frequency * (frame_duration / 1000))
        )
        self.frame_duration = frame_duration
        self.number_observation_samples = int(
            (self.frame_duration / 1000) * self.sampling_frequency
        )
        self.ax_ref.set_title("Reference")
        self.ax_ref.set(
            xlabel="Time (s)",
            ylabel="Amplitude (mV)",
            xlim=(0, self.sample_length / self.sampling_frequency),
            ylim=(0, 1),
        )
        self.fig.suptitle("Heatmmap comparison of " + movement_name, fontsize=14)
        self.x_for_ref = np.linspace(
            0, self.sample_length / self.sampling_frequency, self.ref_data.shape[0]
        )
        self.global_counter = 0
        print("calculating MUAPs with STA ...")
        self.muaps = spikeTriggeredAveraging(
            self.emg_data,
            self.Mu_data,
            0.05,
            self.sample_length,
            self.sampling_frequency,
        )
        print("done")
        self.last_frame = 0
        self.closest_mu = 0

    def make_heatmap_emg(self, frame):
        """
        :return: heatmap of emg data

        """
        heatmap = calculate_emg_rms_grid(
            self.emg_data, frame, self.number_observation_samples
        )
        normalized_heatmap = normalize_2D_array(heatmap)

        # only do the following if +- window size near extrema
        if is_near_extremum(
            frame,
            self.local_maxima,
            self.local_minima,
            time_window=self.frame_duration,
            sampling_frequency=self.sampling_frequency,
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
                    self.heatmaps_flex,
                    np.multiply(normalized_heatmap, 1 / (distance + 0.1)),
                )
                # add heatmap and not normalized heatmap because impact of all heatmaps will be the same but maybe some heatmaps have higher values and i want to use this ??
                # TODO maybe change this (better to use normalized or not ?)
                # IT IS BETER TO USE NORMALIZED BECAUSE IF EMG SCHWANKUNGEN
                self.number_heatmaps_flex += 1
            else:
                self.heatmaps_ex = np.add(
                    self.heatmaps_ex,
                    np.multiply(normalized_heatmap, 1 / (distance + 0.1)),
                )
                self.number_heatmaps_ex += 1

        if self.global_counter == 0:
            hmap = sns.heatmap(
                normalized_heatmap,
                ax=self.ax_emg,
                cmap="RdBu_r",
                cbar=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={"label": "norm. RMS"},
                mask=self.mask_for_heatmap,
            )  # ,cbar_ax=self.ax_emg,)
        else:
            hmap = sns.heatmap(
                normalized_heatmap,
                ax=self.ax_emg,
                cmap="RdBu_r",
                cbar=False,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={"label": "norm. RMS"},
                mask=self.mask_for_heatmap,
            )  # ,cbar_ax=self.ax_emg,)
        self.global_counter += 1
        # self.emg_plot.set(normalized_heatmap, ax=self.ax_emg, cmap='hot', cbar=True, xticklabels=True, yticklabels=True, cbar_kws={'label': 'norm. RMS'},mask=mask_for_heatmap)
        # self.ax_emg.imshow(heatmap, cmap='hot', interpolation='nearest')

    def make_Ref_trackings(self, frame):
        if ("pinch" in self.movement_name) or ("fist" in self.movement_name):
            self.ax_ref.plot(
                self.x_for_ref, normalize_2D_array(self.ref_data[:, 0]), color="blue"
            )
            self.ax_ref.scatter(
                self.x_for_ref[frame],
                normalize_2D_array(self.ref_data[:, 0])[frame],
                color="green",
                marker="x",
                s=90,
                linewidth=3,
            )
        else:
            self.ax_ref.plot(
                self.x_for_ref, normalize_2D_array(self.ref_data), color="blue"
            )
            self.ax_ref.scatter(
                self.x_for_ref[frame],
                normalize_2D_array(self.ref_data)[frame],
                color="green",
                marker="x",
                s=90,
                linewidth=3,
            )

    def make_mu_heatmap(self, frame):
        # search if a spike occured in the current frame and when, in which mu and then display the heatmap of this mu
        # display the mu heatmap of the mu that has a spike most close to current pos
        # mu data = 1,18  -> 1,dann alle spikes in liste
        # closest_mu = self.mu_with_closest_spike(frame)
        closest_mu = self.mu_with_most_spikes_since_last_frame(frame)
        heatmap = calculate_mu_rms_heatmap_for_one_mu(self.muaps[closest_mu])
        normalized_heatmap = normalize_2D_array(heatmap)
        sns.heatmap(
            normalized_heatmap,
            ax=self.ax_MU,
            cmap="RdBu_r",
            cbar=False,
            xticklabels=True,
            yticklabels=True,
            cbar_kws={"label": "norm. RMS"},
        )
        self.ax_MU.set_title("closest MU firing is from: " + str(closest_mu))

    def mu_with_closest_spike(self, frame):
        """
        if there are multiple mu with the same number of spikes, the first one is chosen
        :param frame:
        :return:
        """
        closest_number_of_spikes_away = 10000000000
        closest_mu = 0
        for mu in range(self.Mu_data.shape[1]):
            closest_value = min(
                self.Mu_data[0][mu][0][:], key=lambda value: abs(value - frame)
            )
            distance = abs(frame - closest_value)
            if distance < closest_number_of_spikes_away:
                closest_number_of_spikes_away = distance
                closest_mu = mu
        print("closest mu is: " + str(closest_mu))
        return closest_mu

    def mu_with_most_spikes_since_last_frame(self, frame):
        """
        if there are multiple mu with the same number of spikes, the first one is chosen
        if there are no spikes in the current frame, the last displayed mu is chosen
        :param frame:
        :return:
        """
        most_firings_since_last_frame = 0
        for mu in range(self.Mu_data.shape[1]):
            number_firings = np.count_nonzero(
                (self.Mu_data[0][mu][0][:] > self.last_frame)
                & (self.Mu_data[0][mu][0][:] < frame)
            )
            if number_firings > most_firings_since_last_frame:
                most_firings_since_last_frame = number_firings
                self.closest_mu = mu
        # print("closest mu is: " + str(self.closest_mu))
        # print("it has " + str(most_firings_since_last_frame) + " spikes since last frame")
        self.last_frame = frame
        return self.closest_mu

    def update(self, frame):
        self.ax_emg.clear()
        self.ax_ref.clear()
        self.ax_MU.clear()
        if len(self.fig.axes) > 4:
            self.fig.delaxes(self.fig.axes[4])
            self.ax_emg.relim()
            self.ax_emg.autoscale_view()

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
        self.make_mu_heatmap(frame)
        # Save the frame as an image
        frame_filename = os.path.join(self.path_to_save_plots, f"frame_{frame:03d}.png")
        self.fig.canvas.draw()
        # plt.savefig(frame_filename, bbox_inches='tight', dpi=100)
        pil_image = Image.frombytes(
            "RGB", self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb()
        )
        pil_image.save(frame_filename)
        self.pbar.update(1)

        return self.ax_emg, self.ax_ref, self.ax_MU

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
        # samples = all sample values when using all samples with self.frame_duration in between
        self.samples = np.linspace(
            0, self.sample_length, self.num_samples, endpoint=False, dtype=int
        )
        self.samples = [
            element for element in self.samples if element <= self.ref_data.shape[0]
        ]
        # make both lists to save all coming heatmaps into it by adding the values and dividing at the end through number of heatmaps
        num_rows, num_cols = self.emg_data.shape
        self.heatmaps_flex = np.zeros((num_rows, num_cols))
        self.heatmaps_ex = np.zeros((num_rows, num_cols))
        self.number_heatmaps_flex = 0
        self.number_heatmaps_ex = 0
        if ("pinch" in self.movement_name) or ("fist" in self.movement_name):
            self.local_maxima, self.local_minima = get_locations_of_all_maxima(
                self.ref_data[:, 0]
            )
        else:
            self.local_maxima, self.local_minima = get_locations_of_all_maxima(
                self.ref_data[:]
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
        plt.show()

    def channel_extraction(self, mark_choosen_channels=False):
        self.pbar.close()
        ########
        mean_flex_heatmap = normalize_2D_array(
            np.divide(self.heatmaps_flex, self.number_heatmaps_flex)
        )
        mean_ex_heatmap = normalize_2D_array(
            np.divide(self.heatmaps_ex, self.number_heatmaps_ex)
        )
        difference_heatmap = normalize_2D_array(
            np.abs(np.subtract(mean_ex_heatmap, mean_flex_heatmap))
        )
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
    # movement_list = ["thumb_slow", "thumb_fast", "index_slow", "index_fast", "middle_slow", "middle_fast", "ring_slow",
    #                  "ring_fast", "pinky_slow", "pinky_fast", "fist", "2pinch", "3pinch"]
    # for movement in tqdm.tqdm(movement_list):
    movement = "2pinch"
    heatmap = Heatmap(movement, r"D:\Lab\data\extracted\Sub2")
    heatmap.animate(save=True)

    fps = 2
    heatmap.save_animation(
        r"D:\Lab\MasterArbeit\plots\Heatmaps\emg_animation_"
        + str(movement)
        + str(fps)
        + "fps_most_firings.gif",
        fps=2,
    )
    heatmap.channel_extraction()
    heatmap.channel_extraction(mark_choosen_channels=True)
    # TODO wie erkenne ich wichtige channels ?
    # Kalibrierung :
    #   -> Zeige video von bewegung die sie nachmachen sollen
    #   -> messe nebenzu emg
    #   -> (variable frame_duration)
    #   -> (frame_duration vor peaks in beiden bewegungsenden (flexion/extensoin) sowie bei peaks und frame_duration nach peak anschauen)
    # gehe über alle bewegungszyklen und schaue immer wo regionen mit maximaler aktivität sind (evtl centurion) (alternative zu centurion selber ausdenken)
    # mache 2 listen eine für centurions für bewegungsabschnitt der zu hälfte Extension gehört und eine der zu bewegungshälfte zu flexion zählt
    # liste soll zusätzlich speichern bei wie vielen der bewegungszyklen dieses centurion insgesamt vorkommt
    # WICHTIG !!! manchaml sind genau die bereiche wichtig die zwar nicht rot sind aber die weiß (mittlere Aktivität) sind wichtig,
    # weil diese in einer bewegung aktiv sind und manche nicht
    # evtl als erstes um peaks wie oben ausgeklammert sind schauen ob sichere centurions zu finden die mit aussae über bewegungswechsel (peaks geben)
    # und danach erst die zwischenzyklen ansehen
    # gehe grid für grid und nicht im gesamtkontext weil ansonsten sieht es so aus als ob innervation ähnlich aber eigentlich
    # ist sie einmal am rand von grid 1 und einmal am rand von grid 2 was aber im bild nur 1 pixel weit weg ist
    # evtl. Durchschnittsheatmap machen für beide Bewegungspeaks und dann different heatmap machen (bisschen wie sta ?)
    # für beide peaks sowie alle frame_duration samples davor und danach die heatmaps speichern und dann durchschnitt bilden
    # am ende channel für channel voneinander abziehen

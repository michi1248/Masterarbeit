import os
import pickle
import platform
import socket
import threading
import time
import tkinter as tk
from functools import partial
from pathlib import Path
from tkinter import ttk
from tkinter.font import Font

import numpy as np
import pandas as pd
import vlc


class Realtime_Datagenerator:
    def __init__(self, patient_id: str, recording_time: int, sampling_frequency_emg: int = 2048):
        self.patient_id = patient_id
        # dim = N x 1 x 320 x 192
        self.emg_values = []
        # sampling freq of the emg
        self.sampling_frequency_emg = sampling_frequency_emg
        self.angle_values = []
        # list with start and stop for movements recording in sek
        self.timer = time.time()
        # saves all movement names and the second when emg started recording
        self.values_movie_start_emg = {}
        # to exit the whole emg streaming function
        self.escape_loop = False
        # self.dataset_file = h5py.File(self.path_to_dataset + f"/dataset_{patient_id}.hdf5", "w")
        # to get the emg for a new movement
        self.stop_emg_stream = False
        # dictionary with movement_name: data
        self.emg_list = {}
        # list with all movement names
        self.movement_name = []
        # number of movements we had so far
        self.movement_count = 0
        # time when an emg for a new movement gets captured
        self.emg_time = 0
        # time when a new video starts
        self.video_time = 0
        # difference between emg time and video time for every movement
        self.time_diffs = {}
        # time how long one movement takes to capture
        self.recording_time = recording_time
        # size of one chunk in sample
        self.chunk_size = 64
        ##################################### Stuff for AI/MODEL ##########################################

        ##################################### Stuff for Sockets / Streaming ##########################################
        self.buffer_size = 3
        self.EMG_HEADER = 8
        # Run Biolab Light and select refresh rate = 64
        self.BufferSize = 408 * 64 * 2  # ch, samples, int16 -> 2 bytes
        self.serverIP = "127.0.0.1"  # mit welcher IP verbinden
        self.serverPort = 1234  # mit welchem Port verbinden
        self.emgSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.emgSocket.connect(("127.0.0.1", 31000))
        # self.emgSocket.setblocking(False)
        print("Server is open for connections now.\n")
        # self.emg_client_address = None
        # self.emg_client_address = None
        self.angles_client_address = None

    def createInner(self):
        return Realtime_Datagenerator.PyPlayer(self)

    class PyPlayer(tk.Frame):
        def __init__(self, outer_instance, container, container_instance, title=None):
            tk.Frame.__init__(self, container_instance)
            self.container = container
            self.container_instance = container_instance
            self.initial_directory = Path(os.path.expanduser("~"))
            self.menu_font = Font(family="Verdana", size=20)
            self.default_font = Font(family="Times New Roman", size=16)
            self.timer = 0
            self.times = {}
            self.name = ""
            # create vlc instance
            self.vlc_instance, self.vlc_media_player_instance = self.create_vlc_instance()
            self.outer_Instance = outer_instance
            # main menubar
            self.menubar = tk.Menu(self.container_instance)
            self.menubar.config(font=self.menu_font)

            # vlc video frame
            self.video_panel = ttk.Frame(self.container_instance)
            self.canvas = tk.Canvas(self.video_panel, background="black")
            self.canvas.pack(fill=tk.BOTH, expand=1)
            self.video_panel.pack(fill=tk.BOTH, expand=1)
            self.media_list = [
                str(Folder_For_All_Datasets / "standardized" / (x + "1.avi"))
                for x in [
                    # "wristUpDown_slow",
                    # "wristUpDown_fast",
                    # "wristLeftRight_slow",
                    "wristLeftRight_fast",
                    # "fist_slow",
                    "fist_fast",
                    # "rest_slow",
                    # "thumb_slow",
                    "thumb_fast",
                    # "index_slow",
                    "index_fast",
                    # "middle_slow",
                    "middle_fast",
                    # "ring_slow",
                    "ring_fast",
                    # "pinky_slow",
                    "pinky_fast",
                    # "thumbExtension_slow",
                    # "thumbExtension_fast",
                    # "indexExtension_slow",
                    # "indexExtension_fast",
                    # "middleExtension_slow",
                    # "middleExtension_fast",
                    # "ringExtension_slow",
                    # "ringExtension_fast",
                    # "pinkyExtension_slow",
                    # "pinkyExtension_fast",
                    "rest_fast",
                    # "twoFPinch_slow",
                    "twoFPinch_fast",
                    # "threeFPinch_slow",
                    "threeFPinch_fast",
                    # "pointing_slow",
                    "pointing_fast",
                    # "hardRock_slow",
                    "hardRock_fast",
                    # "peace_slow",
                    "peace_fast",
                ]
            ]
            # self.media_list = self.media_list[:3]

            # controls
            self.create_control_panel()

        def get_times(self):
            return self.times

        def create_control_panel(self):
            """Add control panel."""
            control_panel = ttk.Frame(self.container_instance)

            button = ttk.Button(control_panel, text="start capturing", command=self.start_capturing)
            ttk.Button(control_panel, text="Play", command=self.play)

            # play.pack(side=tk.LEFT)
            button.pack(side=tk.LEFT)
            control_panel.pack(side=tk.BOTTOM)
            self.open()

        def create_vlc_instance(self):
            vlc_instance = vlc.Instance()
            vlc_media_player_instance = vlc_instance.media_player_new()
            self.container_instance.update()
            return vlc_instance, vlc_media_player_instance

        def get_handle(self):
            return self.video_panel.winfo_id()

        def play(self):
            """Play a file."""
            self.outer_Instance.video_time = time.time()
            if not self.vlc_media_player_instance.get_media():
                self.open()
            else:
                if self.vlc_media_player_instance.play() == -1:
                    pass

        def close(self):
            """Close the window."""
            self.container.delete_window()

        def pause(self):
            """Pause the player."""
            self.vlc_media_player_instance.pause()

        def start_capturing(self):
            """Stop the player."""

            time_now = time.time() - self.time

            self.outer_Instance.values_movie_start_emg.update({self.name: time_now})
            self.times.update({self.name: time_now})
            self.timer = time.time()
            # while True:
            #    # print(int(time.time() - self.timer))
            # print("time we calculated:  " + str(int(time.time() - self.timer)))
            # print("video time:  " + str(int(time.time() - self.time)))
            #    if int(time.time() - self.timer) >= 10:     # TODO change
            #        print("here")
            #        break
            print("start of video time:")
            print(time_now)

            time.sleep(self.outer_Instance.recording_time)
            self.vlc_media_player_instance.stop()
            if len(self.media_list) == 0:
                self.close()
                self.outer_Instance.escape_loop = True
            self.outer_Instance.stop_emg_stream = True
            self.open()

        def open(self):
            try:
                self.play_film(self.media_list[0])
            # self.close()
            except Exception:
                pass

        def play_film(self, file):
            """Invokes the `play` method on the vlc instance for the current file."""
            file_name = os.path.basename(file)
            self.name = file_name.split(".avi")[0]
            self.outer_Instance.movement_name.append(self.name)
            self.Media = self.vlc_instance.media_new(file)
            # self.Media.get_meta()
            self.vlc_media_player_instance.set_media(self.Media)
            if platform.system() == "Linux":  # for Linux using the X Server
                self.vlc_media_player_instance.set_xwindow(self.get_handle())
            elif platform.system() == "Windows":  # for Windows
                self.vlc_media_player_instance.set_hwnd(self.get_handle())
            elif platform.system() == "Darwin":  # for MacOS
                self.vlc_media_player_instance.set_nsobject(self.get_handle())
            # self.vlc_media_player_instance.play()
            self.media_list.pop(0)
            self.time = time.time()
            self.play()

        @staticmethod
        def get_film_name(film) -> str:
            """Removes directory from film name."""
            return film.split("/")[-1]

        def create_film_entry(self, film):
            """Adds an entry to the `list_menu` for a given film."""
            self.list_menu.add_command(
                label=self.get_film_name(film),
                command=partial(self.play_film, film),
                font=self.default_font,
            )

    class BaseTkContainer:
        def __init__(self):
            self.tk_instance = tk.Tk()
            self.tk_instance.title("py player")
            self.tk_instance.protocol("WM_DELETE_WINDOW", self.delete_window)
            self.tk_instance.geometry("1920x1080")  # default to 1080p
            self.tk_instance.configure(background="black")
            self.theme = ttk.Style()
            self.theme.theme_use("alt")

        def delete_window(self):
            tk_instance = self.tk_instance
            tk_instance.quit()
            tk_instance.destroy()
            # os._exit(1)

        def __repr__(self):
            return "Base tk Container"

    def get_movies_and_process(self):
        root = self.BaseTkContainer()
        self.PyPlayer(self, root, root.tk_instance, title="pyplayer")
        root.tk_instance.mainloop()
        # root.delete_window()

    def run_parallel(self):
        self.emgSocket.send("startTX".encode("utf-8"))

        t1 = threading.Thread(target=self.get_movies_and_process)
        t2 = threading.Thread(target=self.get_emg)

        t1.start()
        time.sleep(1)
        t2.start()

        # wait till both finishes before continuing main process
        t1.join()

        self.emgSocket.send("stopTX".encode("utf-8"))
        self.emgSocket.close()

        t2.join()

        # get 3d points

        print("Step 1: \t Getting Kinematics Data from Files")

        kinematics_data = {}
        for i in self.movement_name:
            data = pd.read_csv(str(Folder_For_All_Datasets / "3d_coords" / f"{i[:-1]}.csv"))

            # from seconds to samples like this
            start = int((self.values_movie_start_emg[i] * 120))
            # print(start, int((start + (5 * 120))))
            # times 10 because of 10 seconds movement we want to have
            stop = int((start + (self.recording_time * 120)))
            data = data[start:stop]

            # add deviation to every Coord so that it is not absolutely perfect
            # data[timestamp][joint coords]
            # data = np.array(data)
            # for timestamp in range(len(data)):
            #     for coord in range(len(data[timestamp])):
            #         data[timestamp][coord] += np.random.normal(0, coord / 40, 1)

            # print(data.shape)
            skip = int(np.round((64 / 2048) * 120))
            chunk_size__samples = skip * 3

            # print(skip)

            distance_between_chunks__samples = skip
            kinematics_temp = []
            k = 0
            while k + chunk_size__samples <= data.shape[0]:
                to_append = data[k : k + chunk_size__samples].mean(axis=0)

                kinematics_temp.append(to_append)
                k += distance_between_chunks__samples
            kinematics_data[i] = np.array(kinematics_temp)

        print("Step 2: \t Saving Kinematics Data")

        with open("3d_data" + self.patient_id + ".pkl", "wb") as f:
            pickle.dump(kinematics_data, f)

        print("Step 3: \t Getting EMG Data from Memory")

        emg_data = {}
        for k, v in self.emg_list.items():
            # from seconds to samples like this
            start = int((self.values_movie_start_emg[k] - self.time_diffs[k]) * 32)
            # times 10 because of 10 seconds movement we want to have
            stop = int(start + (self.recording_time * 32))
            # print(self.values_movie_start_emg)
            # print("start " + str(start) + "  stop " + str(stop) )
            # print(len(v))
            emg_data[self.movement_name[0]] = np.array([step[None, ...] for step in v[start:stop]])
            self.movement_name.pop(0)

        print("Step 4: \t Saving EMG Data")

        with open("emg_data" + self.patient_id + ".pkl", "wb") as f:
            pickle.dump(emg_data, f)

        return

    # stream emg data and save them into dictionary and after all in -pkl for every movement and speed
    def get_emg(self):
        while True:
            if self.escape_loop:
                break
            save_buffer = []
            print("new video")
            self.emg_time = time.time()
            time_diff = self.emg_time - self.video_time
            while True:
                # print("in emg")
                # exit loop and close emg
                # print(self.stop_emg_stream)
                if self.stop_emg_stream:
                    # print("1212")
                    self.stop_emg_stream = False
                    break
                # get emg data

                try:
                    save_buffer.append(
                        np.frombuffer(self.emgSocket.recv(self.BufferSize), dtype=np.int16).reshape(
                            (408, -1), order="F"
                        )[np.concatenate([np.r_[:64], np.r_[128:384]])]
                    )
                except Exception:
                    continue
            # print("----")
            # print(str(self.movement_count))
            self.time_diffs.update({self.movement_name[self.movement_count]: time_diff})
            print("time diffs:")
            print(self.time_diffs)
            print("length of last emg")
            print(len(save_buffer))
            # self.emg_list[self.movement_name[self.movement_count]] = save_buffer
            self.emg_list.update({self.movement_name[self.movement_count]: save_buffer})
            self.movement_count += 1


if __name__ == "__main__":
    Folder_For_All_Datasets = Path(r"D:\Old Data\Projects\videos_for_data_acquisition\new_hold")

    patient = Realtime_Datagenerator(patient_id="_sub1", sampling_frequency_emg=2048, recording_time=25)
    patient.run_parallel()

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
from exo_controller.Videoplayer import PyPlayer, BaseTkContainer
from exo_controller.Interface_Datageneration_Virtual_Hand import Interface, Container
from exo_controller.exo_controller import Exo_Control


class Realtime_Datagenerator:
    def __init__(
        self,
        patient_id: str,
        recording_time: int,
        sampling_frequency_emg: int = 2048,
        debug=False,
        movements=None,
        grid_order=None,
        use_virtual_hand_interface_for_coord_generation = True
    ):
        if grid_order is None:
            grid_order = [1,2,3,4,5]
        self.grid_order = grid_order
        self.debug = debug
        self.patient_id = patient_id
        # dim = N x 1 x 320 x 192
        self.emg_values = []
        # sampling freq of the emg
        self.sampling_frequency_emg = sampling_frequency_emg
        self.timer = time.time()
        # saves all movement names and the second when emg started recording
        self.values_movie_start_emg = {}
        # to exit the whole emg streaming function
        self.escape_loop = False
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
        self.BufferSize = 408 * 64 * 2  # ch, samples, int16 -> 2 bytes
        # size of one chunk in sample
        self.chunk_size = 64

        self.coords_list_virtual_hand = {}
        self.movement_name_virtual_hand = []
        self.movement_count_virtual_hand = 0
        self.time_diffs_virtual_hand = {}
        self.coords_time = 0
        self.escape_loop_virtual_hand = False
        self.stop_coordinate_stream = False


        if len(self.grid_order) ==5:
            self.emg_indices = np.concatenate([np.r_[:64], np.r_[128:384]])
        else:
            self.emg_indices =  np.r_[128:(128+len(self.grid_order)*64)]
        ##################################### Stuff for Sockets / Streaming ##########################################
        self.buffer_size = 3
        self.EMG_HEADER = 8
        # Run Biolab Light and select refresh rate = 64
        self.BufferSize = 408 * 64 * 2  # ch, samples, int16 -> 2 bytes
        self.serverIP = "127.0.0.1"  # mit welcher IP verbinden
        self.serverPort = 1234  # mit welchem Port verbinden
        self.emgSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.movement_names_videos = movements

        self.use_virtual_hand_interface_for_coord_generation = use_virtual_hand_interface_for_coord_generation

        if not debug:
            self.emgSocket.connect(("127.0.0.1", 31000))
            # self.emgSocket.setblocking(False)
            print("Server is open for connections now.\n")
            # self.emg_client_address = None
            self.emg_client_address = None

    def createInner(self):
        return PyPlayer(self)

    def get_movies_and_process(self):
        if self.use_virtual_hand_interface_for_coord_generation:
            root = Container()
            Interface(self, root, root.tk_instance, title="Interface", movements=self.movement_names_videos, use_virtual_hand_interface_for_coord_generation=self.use_virtual_hand_interface_for_coord_generation)
            root.tk_instance.mainloop()
            # root.delete_window()
        else:
            root = BaseTkContainer()
            PyPlayer(self, root, root.tk_instance, title="pyplayer", movements=self.movement_names_videos)
            root.tk_instance.mainloop()
        # root.delete_window()



    def run_parallel(self):
        if not self.debug:
            self.emgSocket.send("startTX".encode("utf-8"))



        t1 = threading.Thread(target=self.get_movies_and_process)
        if self.use_virtual_hand_interface_for_coord_generation:
            t3 = threading.Thread(target=self.get_coords_virtual_hand_interface)

        t2 = threading.Thread(target=self.get_emg)

        t1.start()
        time.sleep(1)
        t2.start()
        if self.use_virtual_hand_interface_for_coord_generation:
            t3.start()

        # wait till both finishes before continuing main process
        t1.join()

        if not self.debug:
            self.emgSocket.send("stopTX".encode("utf-8"))
            self.emgSocket.close()

        t2.join()


        if self.use_virtual_hand_interface_for_coord_generation:
            t3.join()
        print("dafsd")
        # get 3d points
        print("movie start values:", self.values_movie_start_emg)
        print("Step 1: \t Getting Kinematics Data from Files")

        kinematics_data = {}
        if self.use_virtual_hand_interface_for_coord_generation:

            for k, v in self.coords_list_virtual_hand.items():
                print("coords list key:", k)
                # from seconds to samples like this
                # 60 becuase sampling frequency of the kinematics is 60
                start = int((self.values_movie_start_emg[k] - self.time_diffs_virtual_hand[k]) * 60 )
                # times 10 because of 10 seconds movement we want to have
                stop = int(start + (self.recording_time * 60))

                # emg_data[self.movement_name[0]] = np.array([step[None, ...] for step in v[start:stop]])
                kinematics_data[self.movement_name_virtual_hand[0]] = np.array(v[start:stop])
                self.movement_name_virtual_hand.pop(0)
        else:
            for i in self.movement_name:
                data = pd.read_csv(
                    r"trainings_data/movement_numbers_for_videos/" + str(i) + ".csv"
                )
                data = data.to_numpy()
                # from seconds to samples like this
                start = int((self.values_movie_start_emg[i] * 120))
                # print(start, int((start + (5 * 120))))
                stop = int(
                    (start + (self.recording_time * 120))
                )  # TODO change 120 to variable to also support other fps in the video
                data = data[start:stop]

                # hoch much one incoming emg chunk is in samples in the video
                # e.g hw much is 64 samples in emg in secondds in the video
                # the following lines are necessary to ressample the kinematics data to the emg data
                skip = int(np.round((64 / 2048) * 120))

                # chunk_size__samples = skip * 3
                #
                # distance_between_chunks__samples = skip
                # kinematics_temp = []
                # k = 0
                # while k + chunk_size__samples <= data.shape[0]:
                #     to_append = data[k : k + chunk_size__samples].mean(axis=0)
                #     kinematics_temp.append(to_append)
                #     k += distance_between_chunks__samples
                # kinematics_data[i] = np.array(kinematics_temp)
                kinematics_data[i] = np.array(data)
        print("Step 2: \t Saving Kinematics Data")
        if not self.debug:
            resulting_file = (
                r"trainings_data/resulting_trainings_data/subject_"
                + str(self.patient_id)
                + "/3d_data"
                + ".pkl"
            )
            if not os.path.exists(
                "trainings_data/resulting_trainings_data/subject_"
                + str(self.patient_id)
            ):
                os.makedirs(
                    "trainings_data/resulting_trainings_data/subject_"
                    + str(self.patient_id)
                )

            with open(resulting_file, "wb") as f:
                pickle.dump(kinematics_data, f)

        print("Step 3: \t Getting EMG Data from Memory")

        emg_data = {}
        for k, v in self.emg_list.items():
            print("emg list key:", k)
            # from seconds to samples like this
            # 32 because 2048/64 = 32
            # because one output of the emg is 64 samples and we want ot know how much samples we have to skip in the emg
            # 32 outputs of 64 samples chunk in one second
            start = int((self.values_movie_start_emg[k] - self.time_diffs[k]) * 32)
            # times 10 because of 10 seconds movement we want to have
            stop = int(start + (self.recording_time * 32))

            # emg_data[self.movement_name[0]] = np.array([step[None, ...] for step in v[start:stop]])
            emg_data[self.movement_name[0]] = np.array(v[start:stop])
            self.movement_name.pop(0)

        print("Step 4: \t Saving EMG Data")
        if not self.debug:
            resulting_file = (
                r"trainings_data/resulting_trainings_data/subject_"
                + str(self.patient_id)
                + "/emg_data"
                + ".pkl"
            )
            if not os.path.exists(
                "trainings_data/resulting_trainings_data/subject_"
                + str(self.patient_id)
            ):
                os.makedirs(
                    "trainings_data/resulting_trainings_data/subject_"
                    + str(self.patient_id)
                )

            with open(resulting_file, "wb") as f:
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
                # exit loop and close emg
                # print(self.stop_emg_stream)
                if self.stop_emg_stream:
                    # print("1212")
                    self.stop_emg_stream = False
                    break
                # get emg data

                try:
                    save_buffer.append(
                        np.frombuffer(
                            self.emgSocket.recv(self.BufferSize), dtype=np.int16
                        ).reshape((408, -1), order="F")[self.emg_indices]
                    )
                except Exception:
                    continue
            self.time_diffs.update({self.movement_name[self.movement_count]: time_diff})

            # self.emg_list[self.movement_name[self.movement_count]] = save_buffer
            self.emg_list.update({self.movement_name[self.movement_count]: save_buffer})
            self.movement_count += 1

    def get_coords_virtual_hand_interface(self):
        interface = Exo_Control()
        interface.initialize_all()

        while True:


            if self.escape_loop_virtual_hand:
                break
            save_buffer = []

            self.coords_time = time.time()
            time_diff = self.coords_time - self.video_time
            while True:
                # exit loop and close emg
                if self.stop_coordinate_stream:

                    self.stop_coordinate_stream = False
                    break
                # get emg data

                try:

                    data = interface.get_coords_exo()
                    save_buffer.append(
                        [data[0],data[2]]
                    )
                except Exception as e:
                    print(e)
                    continue
            self.time_diffs_virtual_hand.update({self.movement_name_virtual_hand[self.movement_count_virtual_hand]: time_diff})
            # self.emg_list[self.movement_name[self.movement_count]] = save_buffer
            self.coords_list_virtual_hand.update({self.movement_name_virtual_hand[self.movement_count_virtual_hand]: save_buffer})
            self.movement_count_virtual_hand += 1




if __name__ == "__main__":
    Folder_For_All_Datasets = Path(
        r"D:\Old Data\Projects\videos_for_data_acquisition\new_hold"
    )

    patient = Realtime_Datagenerator(
        debug=True, patient_id="sub1", sampling_frequency_emg=2048, recording_time=5
    )
    patient.run_parallel()

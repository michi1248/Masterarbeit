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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vlc
from exo_controller.Videoplayer import PyPlayer, BaseTkContainer
from exo_controller.Interface_Datageneration_Virtual_Hand import Interface, Container
from exo_controller.exo_controller import Exo_Control
from exo_controller.muovipro import *
from scipy.signal import resample


class Realtime_Datagenerator:
    def __init__(
        self,
        patient_id: str,
        recording_time: int,
        sampling_frequency_emg: int = 2048,
        debug=False,
        movements=None,
        grid_order=None,
        use_virtual_hand_interface_for_coord_generation = True,
        retrain=False,
        retrain_number=0,
        finger_indexes=None,
        use_muovi_pro=False,
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
        self.use_muovi_pro = use_muovi_pro


        self.finger_indexes = finger_indexes

        self.retrain = retrain
        self.retrain_number = retrain_number

        self.coords_list_virtual_hand = {}
        self.movement_name_virtual_hand = []
        self.movement_count_virtual_hand = 0
        self.time_diffs_virtual_hand = {}
        self.coords_time = 0
        self.escape_loop_virtual_hand = False
        self.stop_coordinate_stream = False


        self.recording_started_emg = False
        self.recording_started_virtual_hand = False
        self.time_differences_virtual_hand = {}
        self.time_differences_emg = {}

        if not self.use_muovi_pro:
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
            self.sampling_frequency_emg = 2048
            if not debug:
                self.emgSocket.connect(("127.0.0.1", 31000))
                # self.emgSocket.setblocking(False)
                print("Server is open for connections now.\n")
                # self.emg_client_address = None
                self.emg_client_address = None
                self.number_chunks_one_sample = 64
        else:
            self.sampling_frequency_emg = 2000
            self.number_chunks_one_sample = 18

        self.movement_names_videos = movements

        self.use_virtual_hand_interface_for_coord_generation = use_virtual_hand_interface_for_coord_generation



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
            if self.use_muovi_pro:
                self.emgSocket = Muoviprobe_Interface()
                self.emgSocket.initialize_all()
            else:
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




        t2.join()


        if self.use_virtual_hand_interface_for_coord_generation:
            t3.join()

        if not self.debug:
            if self.use_muovi_pro:
                self.emgSocket.close_connection()
            else:
                self.emgSocket.send("stopTX".encode("utf-8"))
                self.emgSocket.close()
                self.interface.close_connection()

        # get 3d points
        print("movie start values:", self.values_movie_start_emg)
        print("Step 1: \t Getting Kinematics Data from Files")

        kinematics_data = {}
        if self.use_virtual_hand_interface_for_coord_generation:

            for k, v in self.coords_list_virtual_hand.items():
                print("total time between samples: ", np.sum(self.time_differences_virtual_hand[k]))
                print("total samples: ", len(self.time_differences_virtual_hand[k]))

                # from seconds to samples like this
                # # 60 becuase sampling frequency of the kinematics is 60
                # start = int((self.values_movie_start_emg[k] - self.time_diffs_virtual_hand[k]) * 60 )
                # # times 10 because of 10 seconds movement we want to have
                # stop = int(start + (self.recording_time * 60))

                #TODO an dieser Stelle umrechnung für resampling machen
                #am anfang solange erste koordinate wie zeitunterschied mit self.time:diffs_virtual_hand ist
                # danach immer zeit zwischen zwei samples nehmen und dann mit sampling frequency der kinematics multiplizieren
                result_array = []

                mean_time_between_samples = np.mean(self.time_differences_virtual_hand[k])
                mean_fps = 1/mean_time_between_samples
                print("mean fps of coordinates recording: ", mean_fps)

                # we need to know if the kinematics data were first or the emg data
                # make kinematics start - emg start
                # if the result is positive, the kinematics started first
                # if the result is negative, the emg started first
                difference_between_emg_and_kinematics_start = self.time_diffs_virtual_hand[k] - self.time_diffs[k]

                # if the kinematics started first, we need to crop the kinematics os that the start time is the same
                sample_at_which_kinematics_has_same_time_like_start_of_emg = 0
                if difference_between_emg_and_kinematics_start > 0:
                    sum = 0
                    for p in range(len(self.time_differences_virtual_hand[k])):
                        sum += self.time_differences_virtual_hand[k][p]
                        if sum > difference_between_emg_and_kinematics_start:
                            # maximal sum of time differences without being higher than the difference between emg and kinematics start time
                            sample_at_which_kinematics_has_same_time_like_start_of_emg = p-1
                            break
                    #crop the kinematics data, now the kinematics data starts at the same time like the emg data
                    v = v[int(sample_at_which_kinematics_has_same_time_like_start_of_emg):]
                    self.time_differences_virtual_hand[k] = self.time_differences_virtual_hand[k][int(sample_at_which_kinematics_has_same_time_like_start_of_emg):]

                counter= 0
                for time_difference in self.time_differences_virtual_hand[k]:
                    if counter == 0:
                        result_array.append(v[0])
                        counter += 1
                        continue
                    if round(time_difference*self.sampling_frequency_emg) >= 1:
                        number_to_resample = round(time_difference*self.sampling_frequency_emg)
                    else:
                        number_to_resample = 1
                    resampled = resample(v[counter-1:counter],number_to_resample)
                    for value in resampled:
                        result_array.append(value)
                    counter += 1

                # emg_data[self.movement_name[0]] = np.array([step[None, ...] for step in v[start:stop]])
                kinematics_data[self.movement_name_virtual_hand[0]] = np.array(result_array)
                print("length of kinematics data: ", len(result_array))
                print("length of kinematics data in seconds: ", len(result_array)/self.sampling_frequency_emg)
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
                skip = int(np.round((64 / self.sampling_frequency_emg) * 120))

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


        print("Step 2: \t Getting EMG Data from Memory")

        emg_data = {}
        for k, v in self.emg_list.items():

            if self.use_virtual_hand_interface_for_coord_generation:
                print("total time between samples: ", np.sum(self.time_differences_emg[k]))
                print("total samples: ", len(self.time_differences_emg[k]))
                result_array = []

                mean_time_between_samples = np.mean(self.time_differences_emg[k])
                mean_fps = 1 / mean_time_between_samples
                print("mean fps of emg recording: ", mean_fps)

                # we need to know if the kinematics data were first or the emg data
                # make kinematics start - emg start
                # if the result is positive, the kinematics started first
                # if the result is negative, the emg started first
                difference_between_emg_and_kinematics_start = self.time_diffs_virtual_hand[k] - self.time_diffs[k]

                #because for emg the data are send in buffers with 64 length, the first sample of each 64 buffer has other time difference


                print("v shape: ", np.array(v).shape)
                if self.use_muovi_pro:
                    v_reshaped = np.array(
                        v).transpose((1, 0, 2)).reshape(32, -1)
                else:
                    v_reshaped = np.array(
                        v).transpose((1, 0, 2)).reshape(len(self.grid_order) * 64, -1)
                 # afterwards shape of v_reshaped is #channels x #samples
                print("v_reshaped shape: ", v_reshaped.shape)

                time_differences_emg_reshaped = []

                for one_sample in range(v_reshaped.shape[1]):
                    buffer = one_sample // self.number_chunks_one_sample
                    time_difference_this_chunk_split_into_64_samples =  self.time_differences_emg[k][buffer] / self.number_chunks_one_sample
                    time_differences_emg_reshaped.append(time_difference_this_chunk_split_into_64_samples)

                # if the kinematics started first, we need to crop the kinematics os that the start time is the same
                sample_at_which_kinematics_has_same_time_like_start_of_emg = 0
                if difference_between_emg_and_kinematics_start < 0:
                    sum = 0
                    for p in range(len(time_differences_emg_reshaped)):
                        sum += time_differences_emg_reshaped[p]
                        if sum > np.abs(difference_between_emg_and_kinematics_start):
                            # maximal sum of time differences without being higher than the difference between emg and kinematics start time
                            sample_at_which_kinematics_has_same_time_like_start_of_emg = p - 1
                            break
                    # crop the kinematics data, now the kinematics data starts at the same time like the emg data
                    v_reshaped = v_reshaped[:, int(sample_at_which_kinematics_has_same_time_like_start_of_emg):]
                    time_differences_emg_reshaped = time_differences_emg_reshaped[int(sample_at_which_kinematics_has_same_time_like_start_of_emg):]


                v_reshaped = np.array(v_reshaped).transpose()
                print("v_reshaped shape: ", v_reshaped.shape)

                counter = 0
                for time_difference in time_differences_emg_reshaped:
                    if counter == 0:
                        result_array.append(v_reshaped[0])
                        counter += 1
                        continue

                    if round(time_difference * self.sampling_frequency_emg) >= 1:
                        number_to_resample = round(time_difference * self.sampling_frequency_emg)
                    else:
                        number_to_resample = 1
                    resampled = resample(v_reshaped[counter - 1:counter], number_to_resample)
                    for value in resampled:
                        result_array.append(value)
                    counter += 1

                emg_data[self.movement_name[0]] = np.array(result_array).transpose()
                print("emg shape", np.array(result_array).transpose().shape)
                self.movement_name.pop(0)
                print("length of emg data: ", len(result_array))
                print("length of emg data result array in seconds: ", len(result_array) / self.sampling_frequency_emg)


            else:
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

        print("Step 3: \t Saving Kinematics Data")
        if not self.debug:
            adding = ""
            if self.retrain:
                adding = "_retrain" + str(self.retrain_number)
            resulting_file = (
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(self.patient_id)
                    + "/3d_data" + adding + ""
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
            print("number of movements: ", len(kinematics_data))
            print("number of fingers: ", kinematics_data["rest"].shape[1])
            for movement_name, data in kinematics_data.items():
                if data.shape[0] > emg_data[movement_name].shape[1]:
                    kinematics_data[movement_name] = data[:emg_data[movement_name].shape[1],:]

                # plt.figure()
                # plt.plot(data[:,0], "r")
                # plt.plot(data[:,1], "g")
                #
                # for i in range(len(self.time_differences_virtual_hand[movement_name])):
                #     if i == 0:
                #         plt.scatter(i,data[i,0], c="b")
                #         plt.scatter(i,data[i,1], c="b")
                #         continue
                #     plt.scatter(int(np.sum(np.array(self.time_differences_virtual_hand[movement_name][:i]))*2048),self.coords_list_virtual_hand[movement_name][i][0], c="b",marker="x")
                #     plt.scatter(int(np.sum(np.array(self.time_differences_virtual_hand[movement_name][:i])) * 2048), self.coords_list_virtual_hand[movement_name][i][1],
                #                 c="b",marker="x")
                # plt.show()

            with open(resulting_file, "wb") as f:
                pickle.dump(kinematics_data, f)

        print("Step 4: \t Saving EMG Data")
        if not self.debug:
            adding = ""
            if self.retrain:
                adding = "_retrain" + str(self.retrain_number)
            resulting_file = (
                r"trainings_data/resulting_trainings_data/subject_"
                + str(self.patient_id)
                + "/emg_data" + adding + ""
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

            for movement_name, data in emg_data.items():
                if data.shape[1] > kinematics_data[movement_name].shape[0]:
                    emg_data[movement_name] = data[:,:kinematics_data[movement_name].shape[0]]



            with open(resulting_file, "wb") as f:
                pickle.dump(emg_data, f)

        return

    # stream emg data and save them into dictionary and after all in -pkl for every movement and speed
    def get_emg(self):
        while True:
            if self.escape_loop:
                print("exiting outer emg loop")
                break
            save_buffer = []
            time_difference_buffer = []

            print("new video")
            self.emg_time = time.time()
            #time_diff = self.emg_time - self.video_time
            count = 0
            last_time = time.time()
            while True:
                # exit loop and close emg
                # print(self.stop_emg_stream)
                if self.stop_emg_stream:
                    print("exiting inner emg loop")
                    self.stop_emg_stream = False
                    break
                # get emg data

                try:
                    if self.use_muovi_pro:
                        self.emgSocket.clear_socket_buffer()
                        data = self.emgSocket.get_EMG_chunk()
                    else:
                        self.emgSocket.setblocking(0)
                        while True:
                            try:

                                data = self.emgSocket.recv(self.BufferSize)  # Non-blocking receive
                                if not data:
                                    self.emgSocket.setblocking(1)
                                    break  # Break if no more data is in the buffer
                            except BlockingIOError:
                                self.emgSocket.setblocking(1)
                                break  # No more data to read
                        data = np.frombuffer(
                                    self.emgSocket.recv(self.BufferSize), dtype=np.int16
                                ).reshape((408, -1), order="F")[self.emg_indices]
                    if self.recording_started_emg:
                        count += 1
                        if count == 1:
                            # time_diff = measures the time difference between when pressed recording start and when the first sample was taken
                            time_diff = time.time() - self.values_movie_start_emg[
                                self.movement_name[self.movement_count]]
                            time_difference_between_last_sample = 0.0
                            last_time = time.time()
                        else:
                            time_difference_between_last_sample = time.time() - last_time
                            last_time = time.time()

                        save_buffer.append(
                            data
                        )


                        time_difference_buffer.append(time_difference_between_last_sample)
                except Exception as e:
                    print("exception in datastream get emg")
                    print(e)
                    continue
            self.time_diffs.update({self.movement_name[self.movement_count]: time_diff})

            # self.emg_list[self.movement_name[self.movement_count]] = save_buffer
            self.emg_list.update({self.movement_name[self.movement_count]: save_buffer})



            self.time_differences_emg.update(
                {self.movement_name[self.movement_count]: time_difference_buffer})
            self.movement_count += 1

    def get_coords_virtual_hand_interface(self):
        print("in coords")
        self.interface = Exo_Control()
        self.interface.initialize_all()

        while True:

            if self.escape_loop_virtual_hand:
                print("exiting outer coord loop")
                break
            save_buffer = []
            time_difference_buffer = []

            self.coords_time = time.time()
            #time_diff = self.coords_time - self.video_time
            count = 0
            last_time = time.time()
            while True:
                # exit loop and close emg
                if self.stop_coordinate_stream:
                    print("exiting inner coord loop")
                    self.stop_coordinate_stream = False
                    break
                # get emg data

                try:

                    data = self.interface.get_coords_exo()

                    if self.recording_started_virtual_hand:
                        #print(f"fps: {1 / (time.time() - last_time)} Count: {count}")

                        count += 1
                        if count ==1:
                            # time_diff = measures the time difference between when pressed recording start and when the first sample was taken
                            time_diff = time.time() - self.values_movie_start_emg[self.movement_name_virtual_hand[self.movement_count_virtual_hand]]
                            time_difference_between_last_sample = 0.0
                        else:
                            time_difference_between_last_sample = time.time() - last_time
                        last_time = time.time()
                        time_difference_buffer.append(time_difference_between_last_sample)
                        save_buffer.append(
                            [data[finger_index] for finger_index in self.finger_indexes]
                        )
                except Exception as e:
                    print("error in get_coords_virtual_hand_interface")
                    print(e)
                    self.interface.close_connection()
                    continue
            self.time_diffs_virtual_hand.update({self.movement_name_virtual_hand[self.movement_count_virtual_hand]: time_diff})
            # self.emg_list[self.movement_name[self.movement_count]] = save_buffer
            self.coords_list_virtual_hand.update({self.movement_name_virtual_hand[self.movement_count_virtual_hand]: save_buffer})

            self.time_differences_virtual_hand.update({self.movement_name_virtual_hand[self.movement_count_virtual_hand]: time_difference_buffer})
            self.movement_count_virtual_hand += 1




if __name__ == "__main__":
    Folder_For_All_Datasets = Path(
        r"D:\Old Data\Projects\videos_for_data_acquisition\new_hold"
    )
    patient = Realtime_Datagenerator(
        debug=True, patient_id="sub1", sampling_frequency_emg=2048, recording_time=5
    )
    patient.run_parallel()

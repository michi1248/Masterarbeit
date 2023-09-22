from attr import dataclass
import h5py
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch
import socket
import os
from raul_net_v4 import CNNApproximator
import torch
import math
import pickle
from mayavi import mlab
from scipy.io import loadmat
from typing import List, Optional
from multiprocessing import Process
import time
import tkinter as tk
import tkinter as tk
from tkVideoPlayer import TkinterVideo
import os
import time


class Realtime_Datagenerator:
    def __init__(
        self,
        patient_id: str,
        folder_for_all_datasets: str,
        sampling_frequency_emg: int = 2048,
    ):

        self.path_to_dataset = (
            folder_for_all_datasets + "dataset_" + patient_id
        )  # Path(folder_for_all_datasets, patient_id)
        self.folder_for_all_datasets = folder_for_all_datasets
        self.patient_id = patient_id
        # dim = N x 2 x 320 x 192
        self.emg_values = []
        # sampling freq of the emg
        self.sampling_frequency_emg = sampling_frequency_emg
        self.angle_values = []
        # list with start and stop for movements recording in sek
        self.timer = time.time()
        # saves all movement names and the second when emg started recording
        self.values_movie_start_emg = {}

        self.stop_emg_stream = False
        self.emg_list = {}
        self.movement_name = [
            "thumb_fast",
            "thumb_slow",
            "index_fast",
            "index_slow",
            "middle_fast",
            "middle_slow",
        ]
        self.movement_count = 0
        # saves 3 chunks with emg data
        self.queue_emg = [
            np.zeros((2, 320, 64)),
            np.zeros((2, 320, 64)),  # TODO
            np.zeros((2, 320, 64)),
        ]
        # size of one chunk in sample
        self.chunk_size = 64
        # um chunks in the whole emg stream
        self.number_of_chunks_in_record = 1000
        ##################################### Stuff for AI/MODEL ##########################################
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            CNNApproximator.load_from_checkpoint(
                "/Users/erichmarz/Desktop/Bachelorarbeit/RealTimeAI/model.ckpt"
            )
            .to(self.device)
            .eval()  # .to("cuda")
        )
        self.keys = list(
            pd.read_csv(
                "./live-streaming-unity/PythonCode/min_max.csv", index_col=0
            ).index
        )
        # min and max movement angles
        self.min, self.max = (
            pd.read_csv("./live-streaming-unity/PythonCode/min_max.csv", index_col=0)
            .to_numpy()
            .T
        )
        # make folder for patients to save dataset in it
        Path.mkdir(Path(self.path_to_dataset), parents=True, exist_ok=True)
        self.dataset_file = h5py.File(
            self.path_to_dataset + f"/dataset_{patient_id}.hdf5", "w"
        )
        ##################################### Stuff for Sockets / Streaming ##########################################
        self.buffer_size = 3
        self.EMG_HEADER = 8
        # Run Biolab Light and select refresh rate = 64
        self.BufferSize = 408 * 64 * 2  # ch, samples, int16 -> 2 bytes
        self.serverIP = "127.0.0.1"  # mit welcher IP verbinden
        self.serverPort = 1234  # mit welchem Port verbinden
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serverSocket.bind((self.serverIP, self.serverPort))
        self.serverSocket.setblocking(0)
        # self.serverSocket.listen()
        self.IP = "127.0.0.1"
        self.PORT = 31000
        self.emgSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.emgSocket.connect((self.IP, self.PORT))

        # emgSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # emgSocket.bind((serverIP, serverPort))
        # emgSocket.listen()
        # socketsList = [serverSocket, emgSocket]
        print("Server is open for connections now.\n")
        # self.emg_client_address = None
        self.angles_client_address = None
        self.lowpass_sos = butter(
            2,
            20 / (0.5 * 2048),
            btype="low",
            analog=False,
            output="sos",
        )
        self.notch_filter = iirnotch(w0=50, Q=75, fs=self.sampling_frequency_emg)

        ############################### Stuff for myavi Hand #############################
        self.points = mlab.points3d(
            *np.zeros((3, 20)), scale_factor=1, resolution=25, color=(0.15, 0.15, 0.15)
        )
        self.lines = [
            mlab.plot3d(
                *np.zeros((3, 4)), tube_radius=0.225, tube_sides=25, color=color
            )
            for color in [
                (0.09, 0.451, 0.706),
                (1, 0.498, 0.055),
                (0.173, 0.627, 0.173),
                (0.839, 0.133, 0.141),
                (0.592, 0.42, 0.749),
            ]
        ]
        mlab.show()

    def get_movies_and_process(self):
        for filename in os.listdir("./real-time-ai/standardized"):
            self.next_video(filename)

    def run_parallel(self):
        self.emgSocket.send("startTX".encode("utf-8"))

        # start GUI
        # wait for space to start
        # make process for get_emg
        # make process for playing video

        proc = []
        p1 = Process(target=self.get_movies_and_process)
        p2 = Process(target=self.get_emg)
        p1.start()
        proc.append(p1)
        p2.start
        proc.append(p2)
        # wait till both finishes before continuing main process
        for p in proc:
            p.join()

        self.emgSocket.send("stopTX".encode("utf-8"))
        self.emgSocket.close()
        directory = self.folder_for_all_datasets + "dataset_" + self.patient_id + "/"
        # get 3d points
        result = {}
        for i in self.movement_name:
            data = pd.read_csv("./real-time-ai/3d_coords/" + i)
            for axis in ("_x", "_y", "_z"):
                axis_specific_columns = [c for c in data.columns if axis in c]
            data = pd.read_csv(
                "./real-time-ai/3d_coords/" + i, usecols=axis_specific_columns
            )

            # from seconds to samples like this
            start = int((self.values_movie_start_emg[i] * 120))
            # times 10 because of 10 seconds movement we want to have
            stop = int((start + (10 * 120)))
            data = data[start:stop]
            skip = np.mean((64 / 2048) * 120)
            chunks = []

            chunk_size__samples = skip * 3
            distance_between_chunks__samples: skip
            emg_signal_temp = []
            i = 0
            while i + chunk_size__samples <= data.shape[1]:
                emg_signal_temp.append(data[i : i + chunk_size__samples])
                i += distance_between_chunks__samples
            result.update({i: emg_signal_temp})

        with open("3d_data" + self.patient_id + ".pkl", "wb") as f:
            pickle.dump(result, f)

        complete_file = {}

        for k, v in self.emg_list.items():
            self.queue_emg = [
                np.zeros((2, 320, 64)),
                np.zeros((2, 320, 64)),
                np.zeros((2, 320, 64)),
            ]
            mov = []
            # from seconds to samples like this
            start = int(
                (self.values_movie_start_emg[k] * self.sampling_frequency_emg) / 64
            )
            # times 10 because of 10 seconds movement we want to have
            stop = int((start + (10 * self.sampling_frequency_emg)) / 64)
            v = v[start:stop]
            for step in v:
                self.queue_emg.append(step)
                self.queue_emg.pop(0)
                queue_values = np.concatenate(self.queue_emg, axis=-1)
                queue_values = np.reshape(queue_values, (1, 2, 320, 192))
                mov.append(queue_values)
            complete_file.update({self.movement_name[0]: mov})
            self.movement_name.pop(0)

        emg_dataset = self.dataset_file.create_dataset(
            name="emg",
            # 20 beause of 10 movements with 2 speeds each
            # shape=(np.shape(complete_file)),
            compression="lzf",
            data=complete_file,
            # maxshape=(None, 2, 320, self.chunk_size),
            dtype=np.float32,
            # chunks=(1, 2, 320, self.chunk_size),
        )

        return emg_dataset

    def next_video(self, filename):

        root = tk.Tk()
        # root.geometry("900x720")
        title = filename.split(".avi")
        title = title[0]
        self.movement_name.append(title)
        root.title(title)
        button = tk.Button(
            root,
            text="start data capturing",
            command=lambda: self.values_movie_start_emg.update(
                {title: videoplayer.current_duration()}
            ),
        ).pack()
        bbutton = tk.Button(
            root,
            text="next movement",
            command=lambda: videoplayer.stop(),
        ).pack()
        videoplayer = TkinterVideo(master=root, consistant_frame_rate=True, scaled=True)
        videoplayer.load("./real-time-ai/standardized" + "/" + filename)
        videoplayer.pack(expand=True, fill="both")
        videoplayer.play()
        root.mainloop()
        self.stop_emg_stream = True

    # stream emg data and save them into folder in .npy for every movement and speed
    def get_emg(self):
        save_buffer = []
        while True:
            print("running emg stream")
            # exit loop and close emg
            if self.stop_emg_stream == True:
                self.stop_emg_stream = False
                break
            # get emg data

            msg = self.emgSocket.recv(self.BufferSize)
            try:
                encMsg = np.frombuffer(msg, dtype=np.int16).reshape(
                    (408, -1), order="F"
                )
                encMsg = encMsg[
                    np.concatenate([np.r_[:64], np.r_[128:384]])
                ]  # TODO fit to channels
                emg_chunk = filtfilt(
                    *self.notch_filter, x=encMsg, padtype=None, axis=-1
                )

                # make duplicate with 20 Hz Low Pass Filter  and concatenate to original one
                emg_signal = np.stack(
                    np.concatenate(
                        [
                            emg_chunk,
                            sosfiltfilt(self.lowpass_sos, emg_chunk, axis=-1),
                        ],
                        axis=0,
                    )
                )
                save_buffer.append(emg_signal.reshape(2, 320, 64))
            except:
                continue
        dict1 = {self.movement_name[self.movement_count]: save_buffer}
        self.emg_list.update(dict1)
        self.movement_count += 1
        save_buffer = []

    # stream emg data and save them into folder in .npy for every movement and speed
    def get_emg_calibration_data(self):
        # start emg data generation
        # self.emgSocket.send("startTX".encode("utf-8"))
        # print(self.emgSocket.recv(8).decode("utf-8"))
        current_key = "save"
        speed_key = None
        movement_keys = (
            [  # TODO mit dominik absprechen dass er die movements aus min max übernimmt
                "thumb",
                "index",
                "middle",
                "ring",
                "pinky",
                "fist",
                "twoFPinch",
                "threeFPinch",
                "wrist",
                "save",
            ]
        )
        # basic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir + "/CalibrationData/"))
        save_buffer = []
        self.emgSocket.send("startTX".encode("utf-8"))
        while True:
            try:
                bytesAddressPair = self.serverSocket.recvfrom(12)
                message = bytesAddressPair[0].decode("utf-8")
                # print(message)
                # address = bytesAddressPair[1]
                if message == "stop":
                    self.emgSocket.send("stopTX".encode("utf-8"))
                    self.emgSocket.close()
                    save_buffer = []
                    break

                if "_" in message:
                    split = message.split("_")
                    message = split[0]
                    speed_key = split[1]
                if message in movement_keys:
                    if message == "save":
                        save_path = (
                            self.path_to_dataset
                            + "/"
                            + current_key
                            + "_"
                            + speed_key
                            + "_EMG.npy"
                        )
                        data_to_save = np.concatenate(save_buffer, axis=-1)
                        np.save(save_path, data_to_save)
                        print(
                            f"data saved to {save_path}.\n Length of data is {data_to_save.shape}.\n"
                        )
                        save_buffer = []
                    current_key = message
            except Exception as e:
                pass
            msg = self.emgSocket.recv(self.BufferSize)
            try:
                encMsg = np.frombuffer(msg, dtype=np.int16).reshape(
                    (408, -1), order="F"
                )
                encMsg = encMsg[np.concatenate([np.r_[:64], np.r_[128:384]])]
                if current_key != "save":
                    # TODO FILTERN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    save_buffer.append(encMsg)
            except:
                continue

    # streaming emg data giving them into AI and returning output of AI
    def realtime_usage(self):
        last_avg: np.array
        self.angles_client_address = self.waitForUnity()
        self.emgSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.emgSocket.connect((self.IP, self.PORT))
        print(
            f"{self.emgSocket.recv(self.EMG_HEADER).decode('utf-8')} connection for realtime purpose."
        )
        # start emg data generation
        self.emgSocket.send("startTX".encode("utf-8"))
        # basic_path = os.path.abspath(
        #    os.path.join(os.path.dirname(__file__), os.pardir + "/CalibrationData/")
        # )
        while True:
            try:
                # get one emg chunk
                msg = self.emgSocket.recv(self.BufferSize)
                encMsg = np.frombuffer(msg, dtype=np.int16).reshape(
                    (408, -1), order="F"
                )
                # encMsg = encMsg[np.concatenate([np.r_[:64], np.r_[128:384]])]
                encMsg = encMsg[:320]
                # filter part
                # Notch with 50 Hz because of electrical support
                emg_chunk = filtfilt(
                    *self.notch_filter, x=encMsg, padtype=None, axis=-1
                )

                # make duplicate with 20 Hz Low Pass Filter  and concatenate to original one
                emg_signal = np.stack(
                    np.concatenate(
                        [
                            emg_chunk,
                            sosfiltfilt(self.lowpass_sos, emg_chunk, axis=-1),
                        ],
                        axis=0,
                    )
                )
                self.queue_emg.append(emg_signal.reshape(2, 320, 64))
                self.queue_emg.pop(0)

                queue_values = np.concatenate(self.queue_emg, axis=-1)
                queue_values = np.reshape(queue_values, (1, 2, 320, 192))  # TODO
                model_input = torch.from_numpy(queue_values).cuda().float()
                output_npArray = np.clip(
                    self.model(model_input).cpu().detach().numpy(), -1, 1
                )
                # TODO hier numpy array hinzugefügt
                output = np.array(
                    [
                        (output_npArray[0][i] + 1) / 2 * (self.max[i] - self.min[i])
                        + self.min[i]
                        for i, head in enumerate(self.keys)
                    ]
                )
                ######################## Smoothing Barrel Filter ##########################

                if len(last_avg) == 0:
                    last_avg.append(output)

                else:
                    # alpha = learning rate or how much you new value influences the avg the lower the value the smoother
                    alpha = 0.18
                    new_avg = last_avg + (
                        alpha / np.sqrt(np.square((output - np.array(last_avg))))
                    ) * (output - np.array(last_avg))
                    last_avg = new_avg
                ###########################################################################

                # send the AI output to virtual hand to see kinematics
                if not np.isnan(np.min(new_avg)):
                    msg = ""
                    # TODO hier vielleicht anpassen ob new_avg[0]
                    for i in new_avg:
                        msg += str(round(i, 2)) + ","
                    msg = msg[:-1].encode("utf-8")
                    self.serverSocket.sendto(msg, self.angles_client_address)
                    print(msg)

            except KeyboardInterrupt:
                break
        self.emgSocket.send("stopTX".encode("utf-8"))

    @mlab.animate
    def anim(self):
        last_avg: np.array
        self.angles_client_address = self.waitForUnity()
        self.emgSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.emgSocket.connect((self.IP, self.PORT))
        print(
            f"{self.emgSocket.recv(self.EMG_HEADER).decode('utf-8')} connection for realtime purpose."
        )
        # start emg data generation
        self.emgSocket.send("startTX".encode("utf-8"))
        # basic_path = os.path.abspath(
        #    os.path.join(os.path.dirname(__file__), os.pardir + "/CalibrationData/")
        # )
        while True:
            try:
                # get one emg chunk
                msg = self.emgSocket.recv(self.BufferSize)
                encMsg = np.frombuffer(msg, dtype=np.int16).reshape(
                    (408, -1), order="F"
                )
                # encMsg = encMsg[np.concatenate([np.r_[:64], np.r_[128:384]])]
                encMsg = encMsg[:320]
                # filter part
                # Notch with 50 Hz because of electrical support
                emg_chunk = filtfilt(
                    *self.notch_filter, x=encMsg, padtype=None, axis=-1
                )

                # make duplicate with 20 Hz Low Pass Filter  and concatenate to original one
                emg_signal = np.stack(
                    np.concatenate(
                        [
                            emg_chunk,
                            sosfiltfilt(self.lowpass_sos, emg_chunk, axis=-1),
                        ],
                        axis=0,
                    )
                )
                self.queue_emg.append(emg_signal.reshape(2, 320, 64))
                self.queue_emg.pop(0)

                queue_values = np.concatenate(self.queue_emg, axis=-1)
                queue_values = np.reshape(queue_values, (1, 2, 320, 192))  # TODO
                model_input = torch.from_numpy(queue_values).cuda().float()
                output_npArray = np.clip(
                    self.model(model_input).cpu().detach().numpy(), -1, 1
                )
                # TODO hier numpy array hinzugefügt
                output = np.array(
                    [
                        (output_npArray[0][i] + 1) / 2 * (self.max[i] - self.min[i])
                        + self.min[i]
                        for i, head in enumerate(self.keys)
                    ]
                )
                ######################## Smoothing Barrel Filter ##########################

                if len(last_avg) == 0:
                    last_avg.append(output)

                else:
                    # alpha = learning rate or how much you new value influences the avg the lower the value the smoother
                    alpha = 0.18
                    new_avg = last_avg + (
                        alpha / math.sqrt((output - last_avg) ** 2)
                    ) * (output - last_avg)
                    last_avg = new_avg
                ###########################################################################

                (
                    self.points.mlab_source.x,
                    self.points.mlab_source.y,
                    self.points.mlab_source.z,
                ) = prediction

                for line, coordinates in zip(
                    self.lines, np.array_split(prediction, 5, axis=-1)
                ):
                    (
                        line.mlab_source.x,
                        line.mlab_source.y,
                        line.mlab_source.z,
                    ) = coordinates
                yield

            except KeyboardInterrupt:
                break

    def predict_mayavi(self):
        self.anim()
        mlab.show()

    def _split_emg_into_chunks(
        self,
        chunk_size__samples: int,
        distance_between_chunks__samples: int,
        input_emg: np.ndarray,
    ) -> List[np.ndarray]:
        emg_signal_temp = []

        i = 0
        while i + chunk_size__samples <= input_emg.shape[1]:
            emg_signal_temp.append(
                filtfilt(
                    *iirnotch(w0=50, Q=75, fs=self.sampling_frequency_emg),
                    x=input_emg[None, :, i : i + chunk_size__samples],
                    padtype=None,
                    axis=-1,
                )
            )
            i += distance_between_chunks__samples
        return emg_signal_temp

    def _filter_emg(
        input_emg__chunk_list: List[np.ndarray],
        supplementary_filter_sos: Optional[np.ndarray],
    ) -> np.ndarray:
        return np.stack(
            [
                np.concatenate(
                    [chunk, sosfiltfilt(supplementary_filter_sos, chunk, axis=-1)],
                    axis=0,
                )
                for chunk in input_emg__chunk_list
            ]
        ).astype(np.float32)

    def waitForUnity(self):
        unconnected = True
        while unconnected:
            try:
                bytesAddressPair = self.serverSocket.recvfrom(5)
                if not bytesAddressPair:
                    continue
                message = bytesAddressPair[0]
                address = bytesAddressPair[1]
                clientMsg = f"Message from Client {message.decode('utf-8')}"
                clientIP = f"Client IP address {address}"
                print(clientMsg)
                print(clientIP)
                if message.decode("utf-8") == "Unity":
                    unconnected = False
            except:
                continue
        return address

    # get all emg datas for this patient and put together to one hdf file and return this one
    def transform_patient_emg_data(self, patient_id: str):
        directory = self.folder_for_all_datasets + "dataset_" + patient_id + "/"
        complete_file = []
        for dirname in os.listdir(directory):
            if ".npy" in dirname:
                file = np.load(directory + dirname)
                # file = np.reshape(file, (-1, 2, 320, 192))  #TODO
                complete_file.append(file)

            """
            new_dir = directory + dirname
            for file in os.listdir(new_dir):
                if ".npy" in file:
                    file = np.load(new_dir + "/" + file)
                    complete_file.append(file)
            """

        emg_dataset = self.dataset_file.create_dataset(
            name="emg",
            # 20 beause of 10 movements with 2 speeds each
            # shape=(np.shape(complete_file)),
            compression="lzf",
            data=complete_file,
            # maxshape=(None, 2, 320, self.chunk_size),
            dtype=np.float32,
            # chunks=(1, 2, 320, self.chunk_size),
        )
        return emg_dataset


def prediction(patient_id, folder_for_all_datasets, sampling_frequency_emg, mayavi):
    patient = Realtime_Datagenerator(
        patient_id, folder_for_all_datasets, sampling_frequency_emg
    )
    print("------------------------------------------------------")
    print("now starting the predicition process")
    print("------------------------------------------------------")
    if mayavi == False:
        patient.realtime_usage()
    else:
        patient.predict_mayavi()
    patient.emgSocket.close()
    print("")
    print("------------------------------------------------------")
    print("now finishes prediction")
    print("------------------------------------------------------")
    return


def calibration(patient_id, folder_for_all_datasets, sampling_frequency_emg):
    patient = Realtime_Datagenerator(
        patient_id, folder_for_all_datasets, sampling_frequency_emg
    )
    patient.angles_client_address = patient.waitForUnity()
    print("------------------------------------------------------")
    print("now starting the calibration process")
    print("------------------------------------------------------")
    patient.get_emg_calibration_data_mayavi()
    data = patient.transform_patient_emg_data(patient_id)
    print("")
    print("------------------------------------------------------")
    print("finished Calibration")
    print("------------------------------------------------------")
    patient.emgSocket.close()
    return


def both(patient_id, folder_for_all_datasets, sampling_frequency_emg, mayavi):
    patient = Realtime_Datagenerator(
        patient_id, folder_for_all_datasets, sampling_frequency_emg
    )
    patient.angles_client_address = patient.waitForUnity()
    print("------------------------------------------------------")
    print("now starting the calibration process")
    print("------------------------------------------------------")
    patient.get_emg_calibration_data()
    data = patient.transform_patient_emg_data(patient_id)
    # TODO import Method that Trains AI with data
    print("")
    print("------------------------------------------------------")
    print("now starting the Realtime usecase")
    print("------------------------------------------------------")
    if mayavi == False:
        patient.realtime_usage()
    else:
        patient.predict_mayavi()
    patient.emgSocket.close()
    return


if __name__ == "__main__":
    with open("last_variables.txt", "rb") as f:
        last_saves = pickle.load(f)
    last_patient_id = last_saves["last_patient_id"]
    last_sampling_freq = last_saves["last_sampling_freq"]
    folder_for_all_datasets = last_saves["folder_for_all_datasets"]
    f.close
    ###################
    patient = Realtime_Datagenerator(
        last_patient_id, folder_for_all_datasets, last_sampling_freq
    )

    while True:
        while True:
            change_inputs = input("want to change inputs from last run ? (y/n): ")
            if change_inputs == "y" or change_inputs == "n":
                break
            else:
                print("no valid argument, please try again")
        if change_inputs == "n":
            while True:
                mode = input("Please Select Mode: Calibration / Prediction / Both: ")
                if mode == "Calibration" or mode == "Prediction" or mode == "Both":
                    break
                else:
                    print("no valid input please try again")
            if mode == "Both":
                while True:
                    visual = input("Want to visualize with mayavi or unity ? :")
                    if visual == "mayavi":
                        mayavi = True
                        break
                    elif visual == "unity":
                        mayavi = False
                        break
                    else:
                        print("no valid argument")
                both(
                    last_patient_id, folder_for_all_datasets, last_sampling_freq, mayavi
                )
            elif mode == "Calibration":
                calibration(
                    last_patient_id, folder_for_all_datasets, last_sampling_freq
                )
            elif mode == "Prediction":
                while True:
                    visual = input("Want to visualize with mayavi or unity ? :")
                    if visual == "mayavi":
                        mayavi = True
                        break
                    elif visual == "unity":
                        mayavi = False
                        break
                    else:
                        print("no valid argument")
                prediction(
                    last_patient_id, folder_for_all_datasets, last_sampling_freq, mayavi
                )
        else:

            last_patient_id = input(
                "Please type in patient ID (if only want to change patient ID and nothing else type !! at the end of ID: "
            )
            if "!!" not in last_patient_id:
                last_sampling_freq = input("Please type in sampling freq from EMG  ")
                folder_for_all_datasets = input(
                    "Please type in folder for all Datasets "
                )
            vals = {
                "last_patient_id": last_patient_id,
                "last_sampling_freq": last_sampling_freq,
                "folder_for_all_datasets": folder_for_all_datasets,
            }
            file = open("last_variables.txt", "wb")
            pickle.dump(vals, file)
            file.close
            while True:
                mode = input("Please Select Mode: Calibration / Prediction / Both ")
                if mode == "Calibration" or mode == "Prediction" or mode == "Both":
                    break
                else:
                    print("no valid input please try again")
            if mode == "Both":
                while True:
                    visual = input("Want to visualize with mayavi or unity ? :")
                    if visual == "mayavi":
                        mayavi = True
                        break
                    elif visual == "unity":
                        mayavi = False
                        break
                    else:
                        print("no valid argument")
                both(
                    last_patient_id, folder_for_all_datasets, last_sampling_freq, mayavi
                )
            elif mode == "Calibration":
                calibration(
                    last_patient_id, folder_for_all_datasets, last_sampling_freq
                )
            elif mode == "Prediction":
                while True:
                    visual = input("Want to visualize with mayavi or unity ? :")
                    if visual == "mayavi":
                        mayavi = True
                        break
                    elif visual == "unity":
                        mayavi = False
                        break
                    else:
                        print("no valid argument")
                prediction(
                    last_patient_id, folder_for_all_datasets, last_sampling_freq, mayavi
                )
    # patient_id = "001"
    # sampling_freq_emg = 2048
    # folder_for_all_datasets = "../CalibrationData/"
    # main_loop(patient_id, folder_for_all_datasets, sampling_freq_emg)


import socket
import time
import pickle
import numpy as np
import keyboard
import pandas as pd
import torch
from scipy.signal import resample, butter, sosfiltfilt, iirnotch
from scipy.spatial.transform import Rotation
from vispy import app, scene
import ast




class EMG_Interface:
    def __init__(self):
        ###########################################################################
        # Define process for reading data
        self.read_process = None
        self.queue_emg = [np.zeros((1, 320, 64))] * 2
        #self.initialize_all()
        self.process_counter = 0


    def initialize_emg_socket(self):
        """
        initializes class variables needed to connect to EMG and then sets up the connection to the EMG and starts the data transfers
        """
        self.chunk_size = 64
        self.emg_indices = np.concatenate([np.r_[:64], np.r_[128:384]])
        self.buffer_size = 3
        self.EMG_HEADER = 8
        # Run Biolab Light and select refresh rate = 64
        self.BufferSize = 408 * 64 * 2  # ch, samples, int16 -> 2 bytes
        self.connected = False
        self.emgSocket = None

        self.tcp_ip = "localhost"
        self.tcp_port = 31000
        self.sampling_frequency = 2048
        # Wait until socket connects to server
        while not self.connected:
            try:
                self.emgSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.emgSocket.connect((self.tcp_ip, self.tcp_port))

            except Exception as e:
                print(f"EMG open connection: ")
                print(e)
                continue

            # Change status to connected
            self.connected = True

            # Write "startTX" to start the transmission of emg data
            self.write_data("startTX")

            # Print responds from emg server
            print(self.emgSocket.recv(8).decode("utf-8") + " has accepted the connection!")


    def initialize_all(self):
        """

        calls all methods of the class that initialize the needed components.
        By calling this function all necessary steps are to run the exo control.

        """
        self.initialize_filters()
        self.initialize_emg_socket()

    def initialize_filters(self):
        """

        define the filters that are needed to filter emg and movement

        """
        # the filters to use
        self.filter_sos = butter(4, 20, "lowpass", output="sos", fs=2048)
        self.notch_filter = iirnotch(w0=50, Q=75, fs=2044)



    def get_EMG_chunk(self):
        "this method receives one emg chunk from the EMG device and returns it"
        try:
            emg_chunk = \
            np.frombuffer(self.emgSocket.recv(self.BufferSize), dtype=np.int16).reshape((408, -1), order="F")[
                self.emg_indices
            ]

            return emg_chunk.astype(np.float32)

        except Exception as e:

            print("error while receiving emg chunkg")
            return np.empty()

    def close_connection(self):
        """
        This method closes the connection to the EMG device

        """
        # Reset status of connection
        self.connected = False
        # Write message to stop transmission
        self.write_data("stopTX")

        # Close client socket
        self.emgSocket.close()

    def write_data(self, msg):
        """ this method sends data to the emg device"""
        self.emgSocket.send(msg.encode("utf-8"))

    def process(self):
        """

        This method combines all methods. If this Loop is calles to do the exo control in real time
        By pressing "q" the process gets terminated

        """
        max_min_values = None
        while True:
            try:
                chunk = self.get_EMG_chunk()
                if keyboard.read_key() == "p":
                    break
            except Exception as e:
                print(e)
                self.close_connection()
                print("Connection closed")



if __name__ == "__main__":
    pass

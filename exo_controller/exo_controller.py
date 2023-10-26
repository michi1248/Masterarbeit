
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




class Exo_Control:
    def __init__(self):
        ###########################################################################
        # Define process for reading data
        self.read_process = None
        self.queue_emg = [np.zeros((1, 320, 64))] * 2
        #self.initialize_all()
        self.process_counter = 0

    def initialize_exo_socket(self):
        """
        initializes class variables needed to connect to exo and then sets up the connection to the exo

        """
        #self.clients = [1212, 1236]
        self.exoIP = "127.0.0.1"  # mit welcher IP verbinden
        self.exoPort = 1236  # Port von diesem Pc Server
        self.port_from_exo= 1333   #Port von exo client
        #self.connected_to_exo = False
        #while not self.connected_to_exo:
        self.exoSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.exoSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # self.exoSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        while True:
            try:
                self.exoSocket.bind((self.exoIP, self.port_from_exo))
                break
            except Exception as e:
                print(e)
                print("Connection failed")
                self.exoPort +=1
                print("defining new port: ", self.exoPort)
                pass
        self.exoSocket.setblocking(False)
        print("Connection opened to exo skeleton")
        #self.exoSocket.listen()



    def initialize_all(self):
        """

        calls all methods of the class that initialize the needed components.
        By calling this function all necessary steps are to run the exo control.

        """

        self.initialize_exo_socket()


    def move_exo(self,values):
        """
        Input : the values of the positions the exo should go to

        This method sends the handposition to the exo to go into this position.

        """
        #print("Start sending data...\n")
        try:
            # Encode message to utf-8 bytes

            encoded_message = str(values).encode("utf-8")
            #print(encoded_message)

            #  definiere IP  und Port von Client
            print(self.exoSocket.sendto(encoded_message, (self.exoIP, self.exoPort)))
            #print("data sent...\n")
        except Exception as e:
            pass

    def get_force_from_exo(self):
        data, addr = self.exoSocket.recvfrom(1024)
        data = data.decode('utf-8')
        return data

if __name__ == "__main__":
    pass
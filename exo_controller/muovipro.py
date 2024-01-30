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


class Muoviprobe_Interface:
    def __init__(self):
        ###########################################################################

        self.process_counter = 0
        self.frame_len = 18
        self.bytes_in_sample = 2
        self.BufferSize = 38 * self.frame_len * self.bytes_in_sample
        self.emg_indices = np.arange(0,32)

    def _send_configuration_to_device(self) :
        try:
            command_bytes = self._integer_to_bytes(self.command)
            success = self.emgSocket.send(command_bytes)
            return success
        except Exception as e:
            print("error while sending command to device")
            print(e)
            return False

    def _integer_to_bytes(self, command: int) -> bytes:
        return int(command).to_bytes(1, byteorder="big")


    def get_start_command(self) :
        self.command = 1 << 3
        self.command += 0 << 1
        self.command += 1
        return self.command

    def get_stop_command(self) :
        self.command = 1 << 3
        self.command += 0 << 1
        self.command += 0
        return self.command

    def initialize_emg_socket(self):
        """
        initializes class variables needed to connect to EMG and then sets up the connection to the EMG and starts the data transfers
        """

        self.connected = False
        self.emgSocket = None

        self.tcp_ip =  "0.0.0.0" #"192.168.14.1"
        self.tcp_port = 54321
        self.sampling_frequency = 2000
        # Wait until socket connects to server
        while not self.connected:
            try:
                self.this_pc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.this_pc_socket.bind((self.tcp_ip, self.tcp_port))
                self.this_pc_socket.listen(1)
                self.emgSocket, self.address = self.this_pc_socket.accept()

            except Exception as e:
                print(f"cannot open connection: ")
                print(e)

                continue

            # Change status to connected
            self.connected = True
            print("connected to device")
            self.get_start_command()
            self._send_configuration_to_device()




    def initialize_all(self):
        """

        calls all methods of the class that initialize the needed components.
        By calling this function all necessary steps are to run the exo control.

        """

        self.initialize_emg_socket()


    def clear_socket_buffer(self):
        # Make the socket non-blocking
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

    def get_EMG_chunk(self):
        "this method receives one emg chunk from the EMG device and returns it"
        try:
            # emg_chunk = np.frombuffer(
            #     self.emgSocket.recv(self.BufferSize), dtype=np.int16
            # ).reshape((38, -1), order="F")[self.emg_indices]
            emg_chunk = self.emgSocket.recv(self.BufferSize)
            emg_chunk = self._bytes_to_integers(emg_chunk)
            emg_chunk = emg_chunk.reshape((38, -1), order="F")[self.emg_indices]
            print("shape received emg: ", np.shape(emg_chunk))
            return emg_chunk.astype(np.float32)

        except Exception as e:
            print("error while receiving emg chunkg")
            return None

    def _bytes_to_integers(self,data) :
        channel_values = []
        # Separate channels from byte-string. One channel has
        # "bytes_in_sample" many bytes in it.
        for channel_index in range(len(data) // 2):
            channel_start = channel_index * self.bytes_in_sample
            channel_end = (channel_index + 1) * self.bytes_in_sample
            channel = data[channel_start:channel_end]

            # Convert channel's byte value to integer
            value = self._decode_int16(channel)
            channel_values.append(value)

        return np.array(channel_values)

    def _decode_int16(self, bytes_value) :
        value = None
        # Combine 2 bytes to a 16 bit integer value
        value = bytes_value[0] * 256 + bytes_value[1]
        # See if the value is negative and make the two's complement
        if value >= 32768:
            value -= 65536
        return value

    def close_connection(self):
        """
        This method closes the connection to the EMG device

        """
        # Reset status of connection
        self.connected = False
        self.get_stop_command()
        self._send_configuration_to_device()
        # Write message to stop transmission
        # Close client socket
        self.emgSocket.close()
        self.this_pc_socket.close()

    def write_data(self, msg):
        """this method sends data to the emg device"""
        self.emgSocket.send(msg.encode("utf-8"))




if __name__ == "__main__":
    emg_interface = Muoviprobe_Interface()
    emg_interface.initialize_all()
    #emg_interface.clear_socket_buffer()
    while True:
        #emg_interface.clear_socket_buffer()
        emg_chunk = emg_interface.get_EMG_chunk()
        print(emg_chunk)
        time.sleep(0.1)
        # if keyboard.read_key() == "q":
        #     break
    print("closing connection")
    emg_interface.close_connection()
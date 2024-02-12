import socket
import numpy as np
import keyboard


class EMG_Interface:
    def __init__(self, grid_order=None):
        """
        This class is used to connect to the EMG device and to receive the EMG data.

        :param grid_order: list with numbers between 1 and 5
        [1,2,3,4,5] means that we use 5 grids (0:64 and 128:384)
        if less than 5 grids are given, we use the amount of values in the list from multiple in 1 to the number of values
        """
        if grid_order is None:
            grid_order = [1, 2, 3, 4, 5]
        self.grid_order = grid_order

    def initialize_emg_socket(self):
        """
        initializes class variables needed to connect to EMG and then sets up the connection to the EMG and starts the data transfers
        """
        self.chunk_size = 64
        if len(self.grid_order) == 5:
            self.emg_indices = np.concatenate([np.r_[:64], np.r_[128:384]])
        else:
            self.emg_indices = np.r_[128 : (128 + len(self.grid_order) * 64)]

        # size of the buffer in bytes
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
            print(
                self.emgSocket.recv(8).decode("utf-8") + " has accepted the connection!"
            )

    def initialize_all(self):
        """

        calls all methods of the class that initialize the needed components.
        By calling this function all necessary steps are to run the exo control.

        """
        self.initialize_emg_socket()

    def clear_socket_buffer(self):
        """
        This method clears the socket buffer. It is used to remove all old data from the buffer until it is empty
        """
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
            emg_chunk = np.frombuffer(
                self.emgSocket.recv(self.BufferSize), dtype=np.int16
            ).reshape((408, -1), order="F")[self.emg_indices]

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
        """this method sends data to the emg device"""
        self.emgSocket.send(msg.encode("utf-8"))


if __name__ == "__main__":
    pass

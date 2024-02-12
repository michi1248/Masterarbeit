import socket
import numpy as np
import ast


class Exo_Control:
    def __init__(self):
        ###########################################################################
        # Define process for reading data
        self.read_process = None
        self.queue_emg = [np.zeros((1, 320, 64))] * 2
        # self.initialize_all()
        self.process_counter = 0

    def initialize_exo_socket(self):
        """
        initializes class variables needed to connect to exo and then sets up the connection to the exo

        """
        # self.clients = [1212, 1236]
        self.exoIP = "127.0.0.1"  # mit welcher IP verbinden
        self.exoPort = 1236  # Port of exo interface
        self.port_from_exo = 1333  # port from this pc
        # self.connected_to_exo = False
        # while not self.connected_to_exo:
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
                self.exoPort += 1
                print("defining new port: ", self.exoPort)
                pass
        self.exoSocket.setblocking(True)
        print("Connection opened to exo skeleton")
        # self.exoSocket.listen()

    def initialize_all(self):
        """

        calls all methods of the class that initialize the needed components.
        By calling this function all necessary steps are to run the exo control.

        """

        self.initialize_exo_socket()

    def move_exo(self, values):
        """
        Input : the values of the positions the exo should go to

        This method sends the handposition to the exo to go into this position.

        """
        # print("Start sending data...\n")
        try:
            # Encode message to utf-8 bytes

            encoded_message = str(values).encode("utf-8")

            #  definiere IP  und Port von Client
            success = self.exoSocket.sendto(encoded_message, (self.exoIP, self.exoPort))
            if success < 0:
                print("error sending data")
            # print("data sent...\n")
        except Exception as e:
            pass

    def get_coords_exo(self):
        try:
            data = self.exoSocket.recv(1024)
            data = data.decode("utf-8")
            data = ast.literal_eval(data)
            return data

        except Exception as e:
            print(e)
            pass

    def close_connection(self):
        self.exoSocket.close()


if __name__ == "__main__":
    interface = Exo_Control()
    interface.initialize_all()
    while True:
        print(interface.get_coords_exo())

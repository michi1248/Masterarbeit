import socket
import numpy as np

class RealTimeInterface():
    def __init__(self):
        # sampling freq of the emg
        self.sampling_frequency_emg = 2048
        # queue of emg data containing 2 empty chunks
        self.queue_emg = [np.zeros((1, 320, 64))] * 2
        # size of one chunk in sample
        self.chunk_size = 64
        # the indices of the channels to use
        self.emg_indices = np.concatenate([np.r_[:64], np.r_[128:384]])

        self.buffer_size = 3
        self.EMG_HEADER = 8

        # Run Biolab Light and select refresh rate = 64
        self.BufferSize = 408 * 64 * 2  # ch, samples, int16 -> 2 bytes

        self.emgSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.emgSocket.connect(("127.0.0.1", 31000))
        self.emgSocket.send("startTX".encode("utf-8"))
        print(
            f"{self.emgSocket.recv(self.EMG_HEADER).decode('utf-8')} connection for realtime purpose."
        )
        print("Server is open for connections now.\n")

    def on_timer(self, _):
        # Get the emg data from the socket and reshape it to (408, 64) and select the channels of interest
        emg_chunk = np.frombuffer(
            self.emgSocket.recv(self.BufferSize), dtype=np.int16
        ).reshape((408, -1), order="F")[self.emg_indices]

        # Append the new chunk to the queue
        self.queue_emg.append(emg_chunk[None, ...])

        # Concatenate the queue to a single array
        emg_input = np.concatenate(self.queue_emg, axis=-1).astype(np.float32)

        # Compute the RMS of the emg input and reshape it to (8, 8)
        emg_image = np.sqrt(
            np.mean(emg_input[0, 0:64].reshape((8, 8, -1)) ** 2, axis=-1)
        )
        # Normalize the image
        emg_image /= np.max(emg_image)

        # Update the image
        self.image.set_data(emg_image.astype(np.float32))
        self.image.update()

        # Remove the oldest chunk from the queue
        self.queue_emg.pop(0)

    def read_data(self, queue):
        while True:
            # Read emg_chunk from buffer
            try:
                emg_chunk = (
                    np.frombuffer(self.client_socket.recv(self.buffer_size), dtype=np.int16)
                    .reshape((408, -1), order="F")
                    .astype(np.double)
                )
                # Put emg_chunk into the queue which is needed to get access to this data from the main process
                queue.put(emg_chunk)

            except Exception as e:
                print("EMG read data: ")
                print(e)
                continue

    # Function to write data to the emg server: "start/stopTX"
    def write_data(self, msg):
        self.client_socket.send(msg.encode("utf-8"))

    # Open connection to the emg server
    def open_connection(self):
        # Wait until socket connects to server
        while not self.connected:
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.tcp_ip, self.tcp_port))

            except Exception as e:
                print("EMG open connection: ")
                print(e)
                continue

            # Change status to connected
            self.connected = True

            # Write "startTX" to start the transmission of emg data
            self.write_data("startTX")

            # Print responds from emg server
            print(self.client_socket.recv(8).decode("utf-8") + " has accepted the connection!")

        # Define process, set daemon to True (execute process in the background) and start process
        self.read_process = Process(target=self.read_data, args=(self.read_queue,))
        self.read_process.daemon = True
        self.read_process.start()

    # Close connection to the emg
    def close_connection(self):
        # Reset status of connection
        self.connected = False
        # Write message to stop transmission
        self.write_data("stopTX")
        # Terminate process
        self.read_process.terminate()
        # Close client socket
        self.client_socket.close()


    # Clear queue
    def clear_queue(self):
        # Check for queue size, if not 0, get queue item
        while not self.read_queue.qsize() == 0:
            try:
                self.read_queue.get()
            except Exception:
                pass


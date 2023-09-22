import socket

import numpy as np
from vispy import app, scene


class RealTimeInterface:
    def __init__(self):
        # sampling freq of the emg
        self.sampling_frequency_emg = 2048

        # queue of emg data containing 2 empty chunks
        self.queue_emg = [np.zeros((1, 320, 64))] * 2
        # size of one chunk in sample
        self.chunk_size = 64
        # the indices of the channels to use
        self.emg_indices = np.concatenate([np.r_[:64], np.r_[128:384]])

        # #################################### Stuff for Sockets / Streaming ##########################################
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

        # ################################ Vispy ##################################################
        self.canvas = scene.SceneCanvas(
            keys="interactive", bgcolor="white", size=(25, 25), show=True
        )

        self.view = self.canvas.central_widget.add_view()

        self.image = scene.visuals.Image(
            np.zeros((8, 8), dtype=np.float32),
            parent=self.view.scene,
            method="impostor",
            cmap="inferno",
            clim=(0, 1),
            interpolation="lanczos",
        )

        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range()
        self.view.camera.zoom(1)

        self.canvas.measure_fps()

    def start_prediction(self):
        _ = app.Timer(interval=1 / 32, connect=self.on_timer, start=True)
        self.canvas.app.run()

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


if __name__ == "__main__":
    RealTimeInterface().start_prediction()

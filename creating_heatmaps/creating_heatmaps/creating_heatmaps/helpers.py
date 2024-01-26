import pickle
from matplotlib.gridspec import GridSpec
import numpy as np
import os
from tqdm import tqdm

import matplotlib.pyplot as plt


np.set_printoptions(threshold=np.inf)


def load_data(path) -> np.ndarray:
    """Load data from a path."""
    emg = pickle.load(open(os.path.join(path, "emg_data.pkl"), "rb"))
    kinematics = pickle.load(open(os.path.join(path, "3d_data.pkl"), "rb"))
    print(emg.keys())
    print(kinematics.keys())
    return emg, kinematics


def reshape_grid(grid, shape):
    """Reshape a grid to a specific shape."""
    return np.reshape(grid, shape)


def calculate_rms(data, window_size=0.1, sampling_frequency=2048):
    """Calculate the root mean square of a data array."""
    window = int(window_size * sampling_frequency)

    channels, samples = data.shape

    # Pad the data array along the time axis
    padded = np.pad(data, ((0, 0), (window // 2, window // 2)), mode="constant")

    rms = np.zeros((channels, samples))

    for ch in tqdm(range(channels)):
        for i in range(samples):
            rms[ch, i] = np.sqrt(np.mean(padded[ch, i : i + window] ** 2))
    return rms


import cv2
import os


def create_video(images_folder, output_video_path, fps=32):
    image_files = sorted(
        [f for f in os.listdir(images_folder) if f.endswith(".png")],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )

    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(images_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'XVID' for AVI format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each image to the video
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()


def create_heatmaps(
    sampling_frequency,
    frame_len,
    recording_name,
    movement_type,
    frames,
    highest_value,
    ext_grid,
    flex_grid,
    kinematics_data,
    output_path,
):
    x_axis_kinematics = np.linspace(
        0, len(kinematics_data[0]) / sampling_frequency, len(kinematics_data[0])
    )

    kinematics_frame_len = kinematics_data.shape[1] / frames
    buffer_for_rms = []
    buffer_for_x_rms = []

    for i in tqdm(range(frames)):
        time_window = frame_len / sampling_frequency * i
        first_index = i * frame_len
        second_index = i * frame_len + frame_len

        kinematics_index = int(kinematics_frame_len * i)

        fig = plt.figure(figsize=(10, 8))
        fig.suptitle("Time window: " + str(round(time_window, 2)))
        gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        ext_grid_data = (
            np.mean(ext_grid[:, :, first_index:second_index], axis=2) / (highest_value + 1e-10)
        )
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(ext_grid_data, cmap="hot", vmin=0, vmax=1)
        ax1.set_axis_off()
        ax1.set_title("Extensor")

        flex_grid_data = (
            np.mean(flex_grid[:, :, first_index:second_index], axis=2) / (highest_value + 1e-10)
        )
        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(flex_grid_data, cmap="hot", vmin=0, vmax=1)
        ax2.set_axis_off()
        ax2.set_title("Flexor")

        cbar_ax = fig.add_axes(
            [0.15, 0.05, 0.7, 0.02]
        )  # Adjust position and size as needed
        cbar = plt.colorbar(
            ax1.imshow(ext_grid_data, cmap="hot", vmin=0, vmax=1),
            cax=cbar_ax,
            orientation="horizontal",
        )

        # Plot kinematics
        ax4 = plt.subplot(gs[1, :])
        ax4.plot(x_axis_kinematics, kinematics_data[0,:], label="y")
        ax4.plot(x_axis_kinematics, kinematics_data[1,:,], label="z")
        ax4.scatter(
            x_axis_kinematics[kinematics_index],
            kinematics_data[0,kinematics_index],
            c="r",
        )
        ax4.scatter(
            x_axis_kinematics[kinematics_index],
            kinematics_data[1,kinematics_index],
            c="r",
        )
        buffer_for_rms.append(np.mean(ext_grid_data)*11)
        buffer_for_x_rms.append(x_axis_kinematics[kinematics_index])



        ax4.scatter(buffer_for_x_rms,buffer_for_rms)


        plt.savefig(output_path + f"/{recording_name}_{movement_type}_{i}.png")
        plt.close()


def find_bad_channels(movement_type, valid, x_axis, test_emg_data):
    while not valid:
        plt.figure()
        plt.title(f"Input data {movement_type.capitalize()}")
        [
            plt.plot(x_axis, channel / np.max(test_emg_data) + i * 0.1)
            for i, channel in enumerate(test_emg_data)
        ]
        plt.show()

        bad_channels_input = input("Bad channels (ex. false, 5, 10): ")
        input_splits = bad_channels_input.split(",")
        bad_channels = [
            int(channel) for channel in input_splits[1:] if len(input_splits) > 1
        ]
        valid = input_splits[0] == "true"

        # Remove bad channels
        test_emg_data[bad_channels, :] = 0
    return bad_channels

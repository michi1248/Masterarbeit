from creating_heatmaps.creating_heatmaps.creating_heatmaps.helpers import *
import os
from creating_heatmaps.creating_heatmaps.creating_heatmaps.butterworth import Lowpass
from tqdm import tqdm


def main():
    cutting_number = 25000
    sampling_frequency = 2048
    frame_len = 64
    recording_name = "remapped2"
    base_path = r"D:\Lab\MasterArbeit\trainings_data\resulting_trainings_data"
    file_path = rf"subject_Michi_11_01_2024_{recording_name}"
    data_path = os.path.join(base_path, file_path)

    emg, kinematics = load_data(data_path)
    movement_types = emg.keys()

    highest_values = []
    rms_datas = []
    kinematics_datas = []
    for movement_type in movement_types:
        cutting_number= emg[movement_type].shape[1]
        emg_data = emg[movement_type].astype(np.float32)[:,:cutting_number]
        kinematics_data = kinematics[movement_type].astype(np.float32)
        frames = int(cutting_number/frame_len) #emg_data.shape[1]
        #emg_data = np.hstack(emg_data)
        kinematics_data = kinematics_data.T[:,:cutting_number]
        kinematics_datas.append(kinematics_data)
        print("EMG shape: ", emg_data.shape)
        print("Kinematics shape: ", kinematics_data.shape)

        valid = False
        x_axis = np.arange(0, emg_data.shape[1], 1) / sampling_frequency
        test_emg_data = emg_data.copy()
        #bad_channels = find_bad_channels(movement_type, valid, x_axis, test_emg_data)
        bad_channels = [18,42,74,75]

        emg_data[bad_channels, :] = 0

        rms_data = calculate_rms(emg_data)

        highest_value = np.max(rms_data)
        highest_values.append(highest_value)

        # Filter RMS
        lowpass = Lowpass(1, sampling_frequency)
        rms_data = np.array(
            [
                lowpass.filter_channel(channel, multiple_samples=True)
                for channel in rms_data
            ]
        )

        ext_grid, flex_grid = np.split(rms_data, 2, axis=0)
        target_shape = (8, 8, ext_grid.shape[1])

        ext_grid = reshape_grid(ext_grid, target_shape)
        flex_grid = reshape_grid(flex_grid, target_shape)


        rms_datas.append((ext_grid, flex_grid))

    highest_value = np.max(highest_values)
    for i, movement_type in enumerate(movement_types):
        output_path = r"D:\Lab\MasterArbeit\creating_heatmaps\creating_heatmaps\results/" + str(recording_name) + str("/") + str(movement_type)
        # Create folder
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        else:
            # Delete old files
            for file in os.listdir(output_path):
                os.remove(os.path.join(output_path, file))
        ext_grid, flex_grid = rms_datas[i]
        kinematics_data = kinematics_datas[i]
        create_heatmaps(
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
        )

        # Create video
        create_video(
            output_path,
            output_path + f"/{recording_name}_{movement_type}.mp4",
            fps=32,
        )


if __name__ == "__main__":
    main()
    plt.show()

import numpy as np
import pandas as pd
from scipy.signal import resample
from exo_controller.datastream import Realtime_Datagenerator
from exo_controller.helpers import *
from exo_controller.MovementPrediction import MultiDimensionalDecisionTree
from exo_controller.emg_interface import EMG_Interface
from exo_controller.exo_controller import Exo_Control
from exo_controller.filter import MichaelFilter
from exo_controller.grid_arrangement import Grid_Arrangement
from exo_controller.ExtractImportantChannels import ChannelExtraction
from exo_controller import normalizations
from scipy.signal import resample
import keyboard
import torch


def remove_nan_values(data):
    """
    removes nan values from the data
    :param data:
    :return:
    """
    for i in data.keys():
        data[i] = np.array(data[i]).astype(np.float32)
        print("number of nans found: ", np.sum(np.isnan(data[i])))
        data[i][np.isnan(data[i])] = 0

    return data


if __name__ == "__main__":
    use_important_channels = (
        False  # wheather to use only the important channels or every channel
    )
    use_local = True  # whether to use the local model or the time model
    output_on_exo = True  # stream output to exo or print it
    filter_output = True  # whether to filter the output with Bachelor filter or not
    time_for_each_movement_recording = 10  # time in seconds for each movement recording
    load_trained_model = False  # wheather to load a trained model or not
    save_trained_model = False  # wheather to save the trained model or not
    use_spatial_filter = True  # wheather to use a spatial filter on the heatmaps or not
    use_robust_scaling = True  # wheather to use robust scaling or not
    use_mean_subtraction = True  # wheather to use mean subtraction or not (take mean heatmap of the rest movement and subtract it from every emg heatmap)
    grid_order = [1, 2, 3, 4, 5]
    use_recorded_data = False  # r"trainings_data/resulting_trainings_data/subject_Michi_Test2/" # wheather to use recorded data for prediction in realtime or use recorded data (None if want to use realtime data)
    window_size = 150  # window size in ms

    # "Min_Max_Scaling" = min max scaling with max/min is choosen individually for every channel
    # "Robust_all_channels" = robust scaling with q1,q2,median is choosen over all channels
    # "Robust_Scaling"  = robust scaling with q1,q2,median is choosen individually for every channel
    # "Min_Max_Scaling_all_channels" = min max scaling with max/min is choosen over all channels
    scaling_method = "no_scaling"

    # "Gauss_filter" = create and use a gauss filter on heatmaps
    # "Bandpass_filter" = use bandpass filter on raw data and plot the filtered all emg channels
    improvement_methods = ["Gauss_filter", "Bandpass_filter"]
    patient_id = "rest2"
    movements = ["thumb", "index", "2pinch", "rest"]

    if not load_trained_model:
        if not use_recorded_data:
            patient = Realtime_Datagenerator(
                debug=False,
                patient_id=patient_id,
                sampling_frequency_emg=2048,
                recording_time=time_for_each_movement_recording,
            )
            patient.run_parallel()

        resulting_file = (
            r"trainings_data/resulting_trainings_data/subject_"
            + str(patient_id)
            + "/emg_data"
            + ".pkl"
        )
        emg_data = load_pickle_file(resulting_file)
        ref_data = load_pickle_file(
            r"trainings_data/resulting_trainings_data/subject_"
            + str(patient_id)
            + "/3d_data.pkl"
        )
        if use_important_channels:
            # extract the important channels of the grid based on the recorded emg channels with the movement
            important_channels = extract_important_channels_realtime(
                movements, emg_data, ref_data
            )
            print(
                "there were following number of important channels found: ",
                len(important_channels),
            )
            channels = []
            for important_channel in important_channels:
                channels.append(from_grid_position_to_row_position(important_channel))
            print(
                "there were following number of important channels found: ",
                len(channels),
            )
        else:
            channels = range(320)

        normalizer = normalizations.Normalization(
            method=scaling_method,
            grid_order=grid_order,
            important_channels=channels,
            frame_duration=window_size,
        )

        # following lines to resample the ref data to the same length as the emg data since they have different sampling frequencies this is necessary

        for i in emg_data.keys():
            emg_data[i] = np.array(
                emg_data[i].transpose(1, 0, 2).reshape(320, -1)
            )  # reshape emg data such as it has the shape 320 x #samples for each movement
        print("emg data shape: ", emg_data["thumb"].shape)
        # resampling_factor = 2048 / 120
        # for movement in movements:
        #     num_samples_emg = emg_data[movement].shape[1]
        #     num_samples_resampled_ref = int(ref_data[movement].shape[0] * resampling_factor)
        #     resampled_ref = np.empty((num_samples_resampled_ref, 2))
        #     ref_data[movement] = resample(ref_data[movement], num_samples_resampled_ref)
        #
        #     print("length of the resampled ref data: ", len(resampled_ref), file=sys.stderr)
        #     print("length of the emg data: ", num_samples_emg, file=sys.stderr)
        #  # convert emg data to dict with key = movement and value = emg data
        # resample reference data such as it has the same length and shape as the emg data
        ref_data = resample_reference_data(ref_data, emg_data)
        print("ref data shape: ", ref_data["thumb"].shape)

        # remove nan values !! and convert to float instead of int !!
        ref_data = remove_nan_values(ref_data)
        emg_data = remove_nan_values(emg_data)

        # calculate the mean heatmap of the rest movement
        if use_mean_subtraction:
            channel_extractor = ChannelExtraction("rest", emg_data, ref_data)
            mean_rest, _, _ = channel_extractor.get_heatmaps()
            normalizer.set_mean(mean=mean_rest)

        # calculate the normalization values
        normalizer.get_all_emg_data(
            path_to_data=resulting_file,
            movements=["rest", "2pinch", "index", "thumb"],
        )
        normalizer.calculate_normalization_values()

    # initialise the decision/prediction model, build the trainingsdata and train the model

    if not load_trained_model:
        model = MultiDimensionalDecisionTree(
            important_channels=channels,
            movements=movements,
            emg=emg_data,
            ref=ref_data,
            patient_number=patient_id,
            mean_rest=mean_rest,
            normalizer=normalizer,
        )
        model.build_training_data(model.movements)
        # model.load_trainings_data()
        model.save_trainings_data()
        model.train()

        # get the maximal and minimal emg values for each channel over all movements from the trainings data for normalization
        max_emg_value = model.max_value
        min_emg_value = model.min_value

        if save_trained_model:
            model.save_model(subject=patient_id)
            print("model saved")
        best_time_tree = model.evaluate(give_best_time_tree=True)
    else:
        model = MultiDimensionalDecisionTree(
            important_channels=channels,
            movements=movements,
            emg=None,
            ref=None,
            patient_number=patient_id,
        )
        model.load_model(subject=patient_id)
        best_time_tree = 2
        emg_data = load_pickle_file(
            r"trainings_data/resulting_trainings_data/subject_"
            + str(patient_id)
            + "/emg_data.pkl"
        )
        ref_data = load_pickle_file(
            r"trainings_data/resulting_trainings_data/subject_"
            + str(patient_id)
            + "/3d_data.pkl"
        )
        for i in emg_data.keys():
            emg_data[i] = np.array(emg_data[i].transpose(1, 0, 2).reshape(320, -1))
        emg_data = remove_nan_values(emg_data)

        if use_important_channels:
            # extract the important channels of the grid based on the recorded emg channels with the movement
            important_channels = extract_important_channels_realtime(
                movements, emg_data, ref_data
            )
            print(
                "there were following number of important channels found: ",
                len(important_channels),
            )
            channels = []
            for important_channel in important_channels:
                channels.append(from_grid_position_to_row_position(important_channel))
            print(
                "there were following number of important channels found: ",
                len(channels),
            )
        else:
            channels = range(320)

        normalizer = normalizations.Normalization(
            method=scaling_method,
            grid_order=grid_order,
            important_channels=channels,
            frame_duration=window_size,
        )

        # following lines to resample the ref data to the same length as the emg data since they have different sampling frequencies this is necessary

        for i in emg_data.keys():
            emg_data[i] = np.array(
                emg_data[i].transpose(1, 0, 2).reshape(320, -1)
            )  # reshape emg data such as it has the shape 320 x #samples for each movement
        print("emg data shape: ", emg_data["thumb"].shape)

        ref_data = resample_reference_data(ref_data, emg_data)
        print("ref data shape: ", ref_data["thumb"].shape)

        # remove nan values !! and convert to float instead of int !!
        ref_data = remove_nan_values(ref_data)
        emg_data = remove_nan_values(emg_data)

        # calculate the mean heatmap of the rest movement
        if use_mean_subtraction:
            channel_extractor = ChannelExtraction("rest", emg_data, ref_data)
            mean_rest, _, _ = channel_extractor.get_heatmaps()
            normalizer.set_mean(mean=mean_rest)

        # calculate the normalization values
        normalizer.get_all_emg_data(
            path_to_data=r"trainings_data/resulting_trainings_data/subject_"
            + str(patient_id)
            + "/emg_data.pkl",
            movements=["rest", "2pinch", "index", "thumb"],
        )
        normalizer.calculate_normalization_values()

    filter_local = MichaelFilter()
    filter_time = MichaelFilter()
    exo_controller = Exo_Control()
    exo_controller.initialize_all()

    if use_spatial_filter:
        gauss_filter = create_gaussian_filter(size_filter=3)
        grid_aranger = Grid_Arrangement(grid_order)
        grid_aranger.make_grid()

    if use_recorded_data:
        max_chunk_number = np.ceil(
            max(model.num_previous_samples) / 64
        )  # calculate number of how many chunks we have to store till we delete old

        emg_data = load_pickle_file(use_recorded_data + "emg_data.pkl")
        for i in emg_data.keys():
            emg_data[i] = np.array(emg_data[i].transpose(1, 0, 2).reshape(320, -1))
        emg_data = remove_nan_values(emg_data)
        ref_data = load_pickle_file(use_recorded_data + "3d_data.pkl")
        ref_data = remove_nan_values(ref_data)
        ref_data = resample_reference_data(ref_data, emg_data)

        for movement in ref_data.keys():
            emg_buffer = []
            ref_data[movement] = normalize_2D_array(ref_data[movement], axis=0)
            print("movement: ", movement, file=sys.stderr)
            for sample in range(0, 25000, 64):
                chunk = emg_data[movement][:, sample : sample + 64]
                emg_buffer.append(chunk)
                if (
                    len(emg_buffer) > max_chunk_number
                ):  # check if now too many sampels are in the buffer and i can delete old one
                    emg_buffer.pop(0)
                data = np.concatenate(emg_buffer, axis=-1)
                heatmap_local = calculate_emg_rms_row(
                    data, data[0].shape[0], model.window_size_in_samples
                )
                # TODO reshape heatmap into grid shape and subtract mean and normalize

                if use_spatial_filter:
                    heatmap_local = heatmap_local.reshape(320, 1)
                    if len(grid_order) > 2:
                        upper, lower = grid_aranger.transfer_320_into_grid_arangement(
                            heatmap_local
                        )
                        concat = grid_aranger.concatenate_upper_and_lower_grid(
                            upper, lower
                        )
                    else:
                        concat, _ = grid_aranger.transfer_320_into_grid_arangement(
                            heatmap_local
                        )
                    heatmap_local = apply_gaussian_filter(
                        concat, gauss_filter
                    )  # TODO ###################################
                    # TODO ################################### hier überall auch die heatmap wieder reshapen dass 16,27,1 ist und nicht 16,27
                    heatmap_local = heatmap_local.reshape(
                        (heatmap_local.shape[0], heatmap_local.shape[1], 1)
                    )
                    heatmap_local = grid_aranger.transfer_grid_arangement_into_320(
                        heatmap_local
                    )
                    heatmap_local = heatmap_local.reshape(heatmap_local.shape[0])

                res_local = model.trees[0].predict(
                    [heatmap_local]
                )  # result has shape 1,2

                if filter_output:
                    res_local = filter_local.filter(
                        np.array(res_local[0])
                    )  # filter the predcition with my filter from my Bachelor thesis
                else:
                    res_local = np.array(res_local[0])

                previous_heatmap = calculate_emg_rms_row(
                    data,
                    data[0].shape[0] - model.num_previous_samples[best_time_tree - 1],
                    model.window_size_in_samples,
                )

                if use_spatial_filter:
                    previous_heatmap = previous_heatmap.reshape(320, 1)
                    if len(grid_order) > 2:
                        upper, lower = grid_aranger.transfer_320_into_grid_arangement(
                            previous_heatmap
                        )
                        concat = grid_aranger.concatenate_upper_and_lower_grid(
                            upper, lower
                        )
                    else:
                        concat, _ = grid_aranger.transfer_320_into_grid_arangement(
                            previous_heatmap
                        )
                    previous_heatmap = apply_gaussian_filter(
                        concat, gauss_filter
                    )  # TODO ###################################
                    previous_heatmap = previous_heatmap.reshape(
                        (previous_heatmap.shape[0], previous_heatmap.shape[1], 1)
                    )
                    previous_heatmap = grid_aranger.transfer_grid_arangement_into_320(
                        previous_heatmap
                    )
                    previous_heatmap = previous_heatmap.reshape(
                        previous_heatmap.shape[0]
                    )
                # difference_heatmap = normalize_2D_array(np.subtract(heatmap_local, previous_heatmap))
                difference_heatmap = np.subtract(heatmap_local, previous_heatmap)
                if use_spatial_filter:
                    difference_heatmap = difference_heatmap.reshape(320, 1)
                    if len(grid_order) > 2:
                        upper, lower = grid_aranger.transfer_320_into_grid_arangement(
                            difference_heatmap
                        )
                        concat = grid_aranger.concatenate_upper_and_lower_grid(
                            upper, lower
                        )
                    else:
                        concat, _ = grid_aranger.transfer_320_into_grid_arangement(
                            difference_heatmap
                        )
                    difference_heatmap = apply_gaussian_filter(
                        concat, gauss_filter
                    )  # TODO ###################################
                    difference_heatmap = difference_heatmap.reshape(
                        (difference_heatmap.shape[0], difference_heatmap.shape[1], 1)
                    )
                    difference_heatmap = grid_aranger.transfer_grid_arangement_into_320(
                        difference_heatmap
                    )
                    difference_heatmap = difference_heatmap.reshape(
                        difference_heatmap.shape[0]
                    )

                if np.isnan(difference_heatmap).any():
                    res_time = np.array([-1, -1])
                else:
                    res_time = model.trees[best_time_tree].predict([difference_heatmap])
                    if filter_output:
                        res_time = filter_time.filter(
                            np.array(res_time[0])
                        )  # fileter the predcition with my filter from my Bachelor thesis
                    else:
                        res_time = np.array(res_time[0])

                for i in range(2):
                    if res_time[i] > 1:
                        res_time[i] = 1
                    if res_time[i] < 0:
                        res_time[i] = 0
                    if res_local[i] > 1:
                        res_local[i] = 1
                    if res_local[i] < 0:
                        res_local[i] = 0

                if output_on_exo:
                    if use_local:
                        print(
                            "predicted: ",
                            res_local,
                            "  --->     actual: ",
                            ref_data[movement][sample + 64],
                            "             second: ",
                            sample / 2048,
                        )
                        res = [
                            round(res_local[0], 3),
                            0,
                            round(res_local[1], 3),
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                        exo_controller.move_exo(res)
                    else:
                        print(
                            "predicted: ",
                            res_time,
                            "  --->     actual: ",
                            ref_data[movement][sample + 64],
                            "             second: ",
                            sample / 2048,
                        )
                        res = [
                            round(res_time[0], 3),
                            0,
                            round(res_time[1], 3),
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                        exo_controller.move_exo(res)

                else:
                    if use_local:
                        print("local prediction: ", res_local)
                    else:
                        print("time prediction: ", res_time)

    emg_interface = EMG_Interface()
    emg_interface.initialize_all()

    max_chunk_number = np.ceil(
        max(model.num_previous_samples) / 64
    )  # calculate number of how many chunks we have to store till we delete old
    emg_buffer = []

    while True:
        # try:
        chunk = emg_interface.get_EMG_chunk()
        # if chunk.any():
        # continue
        emg_buffer.append(chunk)
        if (
            len(emg_buffer) > max_chunk_number
        ):  # check if now too many sampels are in the buffer and i can delete old one
            emg_buffer.pop(0)
        data = np.concatenate(emg_buffer, axis=-1)
        heatmap_local = calculate_emg_rms_row(
            data, data[0].shape[0], model.window_size_in_samples
        )

        if use_spatial_filter:
            heatmap_local = heatmap_local.reshape(320, 1)
            if len(grid_order) > 2:
                upper, lower = grid_aranger.transfer_320_into_grid_arangement(
                    heatmap_local
                )
                concat = grid_aranger.concatenate_upper_and_lower_grid(upper, lower)
            else:
                concat, _ = grid_aranger.transfer_320_into_grid_arangement(
                    heatmap_local
                )
            heatmap_local = apply_gaussian_filter(
                concat, gauss_filter
            )  # TODO ###################################
            heatmap_local = heatmap_local.reshape(
                (heatmap_local.shape[0], heatmap_local.shape[1], 1)
            )
            heatmap_local = grid_aranger.transfer_grid_arangement_into_320(
                heatmap_local
            )
            heatmap_local = heatmap_local.reshape(heatmap_local.shape[0])

        res_local = model.trees[0].predict([heatmap_local])  # result has shape 1,2

        if filter_output:
            res_local = filter_local.filter(
                np.array(res_local[0])
            )  # fileter the predcition with my filter from my Bachelor thesis
        else:
            res_local = np.array(res_local[0])

        previous_heatmap = calculate_emg_rms_row(
            data,
            data[0].shape[0] - model.num_previous_samples[best_time_tree - 1],
            model.window_size_in_samples,
        )

        if use_spatial_filter:
            previous_heatmap = previous_heatmap.reshape(320, 1)
            if len(grid_order) > 2:
                upper, lower = grid_aranger.transfer_320_into_grid_arangement(
                    previous_heatmap
                )
                concat = grid_aranger.concatenate_upper_and_lower_grid(upper, lower)
            else:
                concat, _ = grid_aranger.transfer_320_into_grid_arangement(
                    previous_heatmap
                )
            previous_heatmap = apply_gaussian_filter(
                concat, gauss_filter
            )  # TODO ###################################
            previous_heatmap = previous_heatmap.reshape(
                (previous_heatmap.shape[0], previous_heatmap.shape[1], 1)
            )
            previous_heatmap = grid_aranger.transfer_grid_arangement_into_320(
                previous_heatmap
            )
            previous_heatmap = previous_heatmap.reshape(previous_heatmap.shape[0])

        # difference_heatmap = normalize_2D_array(np.subtract(heatmap_local, previous_heatmap))
        difference_heatmap = np.subtract(heatmap_local, previous_heatmap)
        if use_spatial_filter:
            difference_heatmap = difference_heatmap.reshape(320, 1)
            if len(grid_order) > 2:
                upper, lower = grid_aranger.transfer_320_into_grid_arangement(
                    difference_heatmap
                )
                concat = grid_aranger.concatenate_upper_and_lower_grid(upper, lower)
            else:
                concat, _ = grid_aranger.transfer_320_into_grid_arangement(
                    difference_heatmap
                )
            difference_heatmap = apply_gaussian_filter(
                concat, gauss_filter
            )  # TODO ###################################
            difference_heatmap = difference_heatmap.reshape(
                (difference_heatmap.shape[0], difference_heatmap.shape[1], 1)
            )
            difference_heatmap = grid_aranger.transfer_grid_arangement_into_320(
                difference_heatmap
            )
            difference_heatmap = difference_heatmap.reshape(difference_heatmap.shape[0])
        if np.isnan(difference_heatmap).any():
            res_time = np.array([-1, -1])
        else:
            res_time = model.trees[best_time_tree].predict([difference_heatmap])
            if filter_output:
                res_time = filter_time.filter(
                    np.array(res_time[0])
                )  # fileter the predcition with my filter from my Bachelor thesis
            else:
                res_time = np.array(res_time[0])

        for i in range(2):
            if res_time[i] > 1:
                res_time[i] = 1
            if res_time[i] < 0:
                res_time[i] = 0
            if res_local[i] > 1:
                res_local[i] = 1
            if res_local[i] < 0:
                res_local[i] = 0

        if output_on_exo:
            if use_local:
                res = [
                    round(res_local[0], 3),
                    0,
                    round(res_local[1], 3),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
                exo_controller.move_exo(res)
            else:
                res = [
                    round(res_time[0], 3),
                    0,
                    round(res_time[1], 3),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
                exo_controller.move_exo(res)

        else:
            if use_local:
                print("local prediction: ", res_local)
            else:
                print("time prediction: ", res_time)

        if keyboard.is_pressed("q"):
            break
        # except Exception as e:
        #     print(e)
        #     emg_interface.close_connection()
        #     print("Connection closed")

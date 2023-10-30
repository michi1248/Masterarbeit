import numpy as np
import pandas as pd
from scipy.signal import resample
from exo_controller.datastream import Realtime_Datagenerator
from exo_controller.helpers import *
from exo_controller.MovementPrediction import MultiDimensionalDecisionTree
from exo_controller.emg_interface import EMG_Interface
from exo_controller.exo_controller import Exo_Control
from exo_controller.filter import MichaelFilter
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
def resample_reference_data(ref_data, emg_data):
    """
    resample the reference data such as it has the same length and shape as the emg data
    :param ref_data:
    :param emg_data:
    :return:
    """
    for movement in ref_data.keys():
        ref_data[movement] = resample(ref_data[movement], emg_data[movement].shape[1], axis=0)
    return ref_data

if __name__ == "__main__":

    use_important_channels = False # wheather to use only the important channels or every channel
    use_local = 1 # whether to use the local model or the time model
    output_on_exo = 1 # stream output to exo or print it
    filter_output = True # whether to filter the output with Bachelor filter or not
    time_for_each_movement_recording = 2 # time in seconds for each movement recording
    load_trained_model = False # wheather to load a trained model or not



    #2 was again after the other tests 1 for different poses after training
    #3 was after that
    patient_id = "sub1"
    movements = ["thumb", "index", "2pinch","rest"]
    if not load_trained_model:
        patient = Realtime_Datagenerator(debug=False, patient_id=patient_id, sampling_frequency_emg=2048, recording_time=time_for_each_movement_recording)
        patient.run_parallel()

        resulting_file = r"trainings_data/resulting_trainings_data/subject_" + str(patient_id) + "/emg_data" + ".pkl"
        emg_data = load_pickle_file(resulting_file)
        ref_data = load_pickle_file(r"trainings_data/resulting_trainings_data/subject_" + str(patient_id) + "/3d_data.pkl")
        # following lines to resample the ref data to the same length as the emg data since they have different sampling frequencies this is necessary


        for i in emg_data.keys():
            emg_data[i] = np.array(emg_data[i].transpose(1,0,2).reshape(320,-1)) # reshape emg data such as it has the shape 320 x #samples for each movement
        print("emg data shape: ", emg_data["thumb"].shape)
        resampling_factor = 2048 / 120
        for movement in movements:
            num_samples_emg = emg_data[movement].shape[1]
            num_samples_resampled_ref = int(ref_data[movement].shape[0] * resampling_factor)
            resampled_ref = np.empty((num_samples_resampled_ref, 2))
            ref_data[movement] = resample(ref_data[movement], num_samples_resampled_ref)

            print("length of the resampled ref data: ", len(resampled_ref), file=sys.stderr)
            print("length of the emg data: ", num_samples_emg, file=sys.stderr)
         # convert emg data to dict with key = movement and value = emg data
        # resample reference data such as it has the same length and shape as the emg data
        ref_data = resample_reference_data(ref_data, emg_data)

        # remove nan values !! and convert to float instead of int !!
        ref_data = remove_nan_values(ref_data)
        emg_data = remove_nan_values(emg_data)


    # here : emg data shape = #samples,1,320,64 mit einem dict für jede movement
    # ref data shape = #samples,2  dict für jede movement

    if use_important_channels:
        #extract the important channels of the grid based on the recorded emg channels with the movement
        important_channels = extract_important_channels_realtime(movements, emg_data, ref_data)
        print("there were following number of important channels found: ",len(important_channels))
        channels = []
        for important_channel in important_channels:
            channels.append(from_grid_position_to_row_position(important_channel))
        print("there were following number of important channels found: ",len(channels))
    else:
        channels = range(320)


    #initialise the decision/prediction model, build the trainingsdata and train the model

    if not load_trained_model:
        model = MultiDimensionalDecisionTree(important_channels=channels, movements=movements, emg=emg_data,
                                             ref=ref_data, patient_number=patient_id)
        model.build_training_data(model.movements)
        #model.load_trainings_data()
        model.save_trainings_data()
        model.train()
        model.save_model(subject=patient_id)
        best_time_tree = model.evaluate(give_best_time_tree=True)
    else:
        model = MultiDimensionalDecisionTree(important_channels=channels, movements=movements, emg=None,
                                             ref=None, patient_number=patient_id)
        model.load_model(subject=patient_id)
        best_time_tree = 2

    filter_local = MichaelFilter()
    filter_time = MichaelFilter()



    exo_controller = Exo_Control()
    exo_controller.initialize_all()
    emg_interface = EMG_Interface()
    emg_interface.initialize_all()

    max_chunk_number = np.ceil(max(model.num_previous_samples) / 64)  # calculate number of how many chunks we have to store till we delete old
    emg_buffer = []

    while True:
        #try:
        chunk = emg_interface.get_EMG_chunk()
        #if chunk.any():
            #continue
        emg_buffer.append(chunk)
        if len(emg_buffer) > max_chunk_number:  # check if now too many sampels are in the buffer and i can delete old one
            emg_buffer.pop(0)
        data = np.concatenate(emg_buffer, axis=-1)
        heatmap_local = calculate_emg_rms_row(data, data[0].shape[0], model.window_size_in_samples)
        heatmap_local = normalize_2D_array(heatmap_local)
        res_local = model.trees[0].predict([heatmap_local]) # result has shape 1,2

        if filter_output:
            res_local = filter_local.filter(np.array(res_local[0]))  # fileter the predcition with my filter from my Bachelor thesis
        else:
            res_local = np.array(res_local[0])

        previous_heatmap = calculate_emg_rms_row(data, data[0].shape[0] - model.num_previous_samples[best_time_tree-1],model.window_size_in_samples)
        previous_heatmap = normalize_2D_array(previous_heatmap)
        difference_heatmap = normalize_2D_array(np.subtract(heatmap_local, previous_heatmap))
        if np.isnan(difference_heatmap).any():
            res_time = np.array([-1, -1])
        else:
            res_time = model.trees[best_time_tree].predict([difference_heatmap])
            if filter_output:
                res_time = filter_time.filter(np.array(res_time[0]))  # fileter the predcition with my filter from my Bachelor thesis
            else:
                res_time = np.array(res_time[0])

        for i in range(2):
            if res_time[i]> 1:
                res_time[i] = 1
            if res_time[i] < 0:
                res_time[i] = 0
            if res_local[i] > 1:
                res_local[i] = 1
            if res_local[i] < 0:
                res_local[i] = 0

        if output_on_exo:
            if use_local:
                res = [round(res_local[0], 3), 0, round(res_local[1], 3), 0, 0, 0, 0, 0, 0]
                exo_controller.move_exo(res)
            else:
                res = [round(res_time[0], 3), 0, round(res_time[1], 3), 0, 0, 0, 0, 0, 0]
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








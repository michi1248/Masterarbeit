import numpy as np
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
    #play videos and patient mimics the movements
    #parallel meassure the emg
    #then generate the trainings dataset
    patient_id = "sub1"
    patient = Realtime_Datagenerator(debug=False, patient_id=patient_id, sampling_frequency_emg=2048, recording_time=1)
    patient.run_parallel()

    movements = ["thumb", "index", "2pinch"]
    resulting_file = r"trainings_data/resulting_trainings_data/subject_" + str(patient_id) + "/emg_data" + ".pkl"
    emg_data = load_pickle_file(resulting_file)
    ref_data = load_pickle_file(r"trainings_data/resulting_trainings_data/subject_" + str(patient_id) + "/3d_data.pkl")

    #reshape the emg data

    for i in emg_data.keys():
        emg_data[i] = emg_data[i].transpose(1,0,2).reshape(320,-1)
    print("emg data shape: ", emg_data["thumb"].shape)
    # resample reference data such as it has the same length and shape as the emg data
    ref_data = resample_reference_data(ref_data, emg_data)

    # remove nan values and convert to float instead of int
    ref_data = remove_nan_values(ref_data)
    emg_data = remove_nan_values(emg_data)


    # here : emg data shape = #samples,1,320,64 mit einem dict für jede movement
    # ref data shape = #samples,2  dict für jede movement

    #TODO reshape emg data and ref data into the right format ===> movement_number x 320 x  length aufnahme in samples

    #extract the important channels of the grid based on the recorded emg channels with the movement
    # important_channels = extract_important_channels_realtime(movements, emg_data, ref_data)
    # print("there were following number of important channels found: ",len(important_channels))
    # channels = []
    # for important_channel in important_channels:
    #     channels.append(from_grid_position_to_row_position(important_channel))
    # print("there were following number of important channels found: ",len(channels))


    #initialise the decision/prediction model, build the trainingsdata and train the model
    channels = range(320)
    model = MultiDimensionalDecisionTree(important_channels=channels, movements=movements, emg=emg_data, ref=ref_data,patient_number=patient_id)
    model.build_training_data(model.movements)
    #model.load_trainings_data()
    model.save_trainings_data()
    model.train()
    best_time_tree = model.evaluate(give_best_time_tree=True)
    filter_local = MichaelFilter()
    filter_time = MichaelFilter()



    exo_controller = Exo_Control()
    exo_controller.initialize_all()
    emg_interface = EMG_Interface()
    emg_interface.initialize_all()

    max_chunk_number = np.ceil(max(model.num_previous_samples) / 64)  # calculate number of how many chunks we have to store till we delete old
    emg_buffer = []

    while True:
        try:
            chunk = emg_interface.get_EMG_chunk()
            emg_buffer.append(chunk)
            if len(emg_buffer) > max_chunk_number:  # check if now too many sampels are in the buffer and i can delete old one
                emg_buffer.pop(0)
            data = np.concatenate(emg_buffer, axis=-1)
            heatmap_local = calculate_emg_rms_row(data, data[0].shape[0], model.window_size_in_samples)
            heatmap_local = normalize_2D_array(heatmap_local)
            res_local = model.trees[0].predict([heatmap_local]) # result has shape 1,2

            prediction_local = filter_local.filter(np.array(res_local[0]))  # fileter the predcition with my filter from my Bachelor thesis

            previous_heatmap = calculate_emg_rms_row(data, data[0].shape[0] - model.num_previous_samples[best_time_tree-1],model.window_size_in_samples)
            previous_heatmap = normalize_2D_array(previous_heatmap)
            difference_heatmap = normalize_2D_array(np.subtract(heatmap_local, previous_heatmap))
            if np.isnan(difference_heatmap).any():
                res_time = np.array([[-1, -1]])
                prediction_time = np.array([[-1, -1]])
            else:
                res_time = model.trees[best_time_tree].predict([difference_heatmap])
                prediction_time = filter_time.filter(np.array(res_time[0]))  # fileter the predcition with my filter from my Bachelor thesis

            if prediction_time > 1:
                prediction_time = 1
            if prediction_time < 0:
                prediction_time = 0

            if prediction_local > 1:
                prediction_local = 1
            if prediction_local < 0:
                prediction_local = 0

            #res = [round(input[0], 3), 0, round(input[1], 3), 0, 0, 0, 0, 0, 0]
            #exo_controller.move_exo(res)
            print("before: " + str(res_local) + "  " + str(res_time)+ "  " )
            print("after: " + str(prediction_local) + "  " + str(prediction_time) + "  ")
            if keyboard.is_pressed("q"):
                break
        except Exception as e:
            print(e)
            emg_interface.close_connection()
            print("Connection closed")

    #TODO filter einführen für letzte predictions von model und diese zusätzlich smoothen






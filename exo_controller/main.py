from datastream import Realtime_Datagenerator
from helpers import *
from ExtractImportantChannels import extract_important_channels, from_grid_position_to_row_position
from MovementPrediction import MultiDimensionalDecisionTree
from emg_interface import EMG_Interface
from exo_controller import Exo_Control
import keyboard


if __name__ == "__main__":
    #play videos and patient mimics the movements
    #parallel meassure the emg
    #then generate the trainings dataset
    patient_id = "sub1"
    patient = Realtime_Datagenerator(debug=True, patient_id=patient_id, sampling_frequency_emg=2048, recording_time=5)
    patient.run_parallel()

    movements = ["thumb_slow", "index_slow", "2pinch"]
    resulting_file = r"../trainings_data/resulting_trainings_data/subject_" + str(patient_id) + "/emg_data" + ".pkl"
    emg_data = load_pickle_file(resulting_file)
    ref_data = load_pickle_file(r"../trainings_data/resulting_trainings_data/subject_" + str(patient_id) + "/3d_data.pkl")

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
    # model.load_trainings_data()
    model.save_trainings_data()
    model.train()
    model.evaluate()


    #emg interface
    emg_interface = EMG_Interface()
    emg_interface.initialize_all()
    exo_controller = Exo_Control()
    exo_controller.initialize_all()
    while True:
        try:
            chunk = emg_interface.get_EMG_chunk()
            # TODO chunk is in the shape of 1 x 320 x 192 ( has last 2 and new  64 samples chunk in it ==> so i have to remove the old ones)
            #TODO liste mit alten EMG werten und predict muss unterschiedliche inputs für unterschiedliche bäume erhalten
            input = model.predict(chunk,[0,1,2,3,4])
            res = [round(input[0], 3), 0, round(input[1], 3), 0, 0, 0, 0, 0, 0]
            exo_controller.move_exo(res)
            if keyboard.read_key() == "p":
                break
        except Exception as e:
            print(e)
            emg_interface.close_connection()
            print("Connection closed")

    #TODO filter einführen für letzte predictions von model und diese zusätzlich smoothen



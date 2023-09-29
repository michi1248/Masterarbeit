from datastream import Realtime_Datagenerator
from helpers import *
from ExtractImportantChannels import extract_important_channels, from_grid_position_to_row_position
from MovementPrediction import MultiDimensionalDecisionTree


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
    important_channels = extract_important_channels_realtime(movements, emg_data, ref_data)
    print("there were following number of important channels found: ",len(important_channels))
    channels = []
    for important_channel in important_channels:
        channels.append(from_grid_position_to_row_position(important_channel))
    print("there were following number of important channels found: ",len(channels))


    #initialise the decision/prediction model, build the trainingsdata and train the model
    channels = range(320)
    model = MultiDimensionalDecisionTree(important_channels=channels, movements=movements)
    model.build_training_data(model.movements, r"D:\Lab\data\extracted\Sub2")
    # odel.load_trainings_data()
    model.save_trainings_data()
    model.train()
    model.evaluate()

    #TODO trainigsdataset bauen (Videos einfach nehmen aber für 3d coords nur die wichtigen coords nehmen )

    #TODO in emg_interface dauerhaft kommunikaiton mit emg aufbauen
    #dann immmer einen chunk emg empfangen und diesen reshapen / times berechnen etc in model prediction geben
    #TODO prediction function for one input in model
    #TODO filter einführen für letzte predictions von model und diese zusätzlich smoothen
    #TODO kommunikation  mit exo einführen



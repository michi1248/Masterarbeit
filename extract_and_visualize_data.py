import os
import pickle
import scipy.io as sio
import tqdm
import pandas
import numpy as np
import matplotlib.pyplot as plt

def extract_mat_to_pkl(path_to_save ,emg_folder , MU_folder, movement_ground_truth_folder, attributes_to_open = None):
    movement_list = ["thumb_slow","thumb_fast", "index_slow", "index_fast","middle_slow", "middle_fast", "ring_slow", "ring_fast", "pinky_slow", "pinky_fast", "fist", "2pinch", "3pinch"]
    for subject in os.listdir(emg_folder):
        # subject = sub1 , sub2 , sub3 ...
        # create output folder
        output_folder = os.path.join(path_to_save,subject)

        if (not "DATATABLES" in subject) and (not "Ordner_Struktur" in subject):
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            data_dict = {}

            #extract emg and MU data
            subject_folder = os.path.join(emg_folder, subject)
            subject_folder= os.path.join(subject_folder,"TaskMUData")
            for movement in tqdm.tqdm(os.listdir(subject_folder), desc='extracting emg data for ' + subject + '...'):
                # movement =Sub1_1.mat, Sub1_2.mat
                movement_file_path = os.path.join(subject_folder, movement)
                #get movement name by extracting movement number and looking it up in movement_list
                movement_name = movement_list[int(movement.split("_")[1].split(".")[0])-1]

                #load mat file
                mat_data = sio.loadmat(movement_file_path)
                #save data in dict
                data_dict[movement_name] = {key: mat_data[key] for key in attributes_to_open}

            # extract ref data
            subject_name = subject.split("Sub")[1]
            ref_file_path = os.path.join(movement_ground_truth_folder,
                                         "Exp" + subject_name + "_B_DATATABLE.mat")
            # get movement ref file for this subject
            # load mat file
            mat_data = sio.loadmat(ref_file_path)
            # save data in dict
            data_dict["ref"] = mat_data["DATATABLE"][0]

            #save emg data as pickle
            output_file_path = os.path.join(output_folder, "data.pickle")
            with open(output_file_path, 'wb') as f:
                pickle.dump(data_dict, f)
    return

def open_all_files_for_one_patient_and_movement(folder_path, movement_name):
    movement_list = ["thumb_slow", "thumb_fast", "index_slow", "index_fast", "middle_slow", "middle_fast", "ring_slow",
                     "ring_fast", "pinky_slow", "pinky_fast", "fist", "2pinch", "3pinch"]
    movement_number = movement_list.index(movement_name)+1
    file_path = os.path.join(folder_path, "data.pickle")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    emg_data = data[movement_name]["SIG"]
    MU_data = data[movement_name]["MUPulses"]
    ref_file = os.path.join(folder_path,  str(movement_number) + "_transformed.csv")
    ref_data = pandas.read_csv(ref_file)

    return emg_data,MU_data,ref_data

def visualize_data(emg_data,MU_data,ref_data):
    # plot emg
    plt.figure()
    # 4,5 = row/col in of channels of all grids [0,:] = all samples
    plt.plot(emg_data[4,5][0,:])
    plt.title("emg")
    plt.show()

    # plot MU
    plt.figure()
    max_columns = max(arr.shape[1] for arr in MU_data[0,:])

    # Create an empty array to store the stacked arrays
    stacked_array = np.empty((len(MU_data[0,:]), max_columns), dtype=object)

    # Fill the empty array with the arrays from the list
    count = 0
    for i in MU_data[0,:]:
        stacked_array[count, :i.shape[1]] = i[0,:]
        count += 1

    signal_length = emg_data[0][1][0].shape[0]
    result_array = np.zeros((18,signal_length))
    for MU in range(18):
        for pulse in stacked_array[MU]:
            if pulse != None:
                result_array[MU,pulse] = 1
    plot_spiketrain(result_array)


    # plot ref
    plt.figure()
    plt.plot(ref_data)
    plt.title("ref")
    plt.show()

    return

def plot_spiketrain(spike_array):
    """plot spiketrain of MUAPs with input is a 2D array with MUAPs as rows and spiketime as columns
        the spiketime has length of the signal length consisting of 0s and for every sample that has a spike a 1"""
    plt.figure()
    for MU in range(18):
        count = 0
        for spike in spike_array[MU]:
            if(spike != 0):
                plt.vlines(x =count , ymin=MU, ymax=MU+1)
            count += 1
    plt.title("MU")
    plt.show()

if __name__ == "__main__":
    # sync force, emg and MUAP
    # g1, g2, g3, g4, g5 = grid 1 to 5
    # EmgDataOfOnePatient = r"D:\Lab\data\Data_Marius" #\Sub2\TaskMUData"
    # MU_data_of_patient = r"D:\Lab\data\Data_Marius" #\Sub2\TaskMUData"
    # attributes_to_open = ['MUPulses', 'SIG']
    # path_of_movement_ref = r"D:\Lab\data\Data_Marius\DATATABLES"
    # path_to_save = r"D:\Lab\data\extracted"

    #extract_mat_to_pkl(path_to_save= path_to_save,emg_folder= EmgDataOfOnePatient , MU_folder=MU_data_of_patient , movement_ground_truth_folder= path_of_movement_ref,attributes_to_open=attributes_to_open)

    emg_data,MU_data,ref_data = open_all_files_for_one_patient_and_movement(folder_path= r"D:\Lab\data\extracted\sub2_Charly", movement_name= "index_slow")
    visualize_data(emg_data,MU_data,ref_data)





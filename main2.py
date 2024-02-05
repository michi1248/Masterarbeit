import numpy as np
import pandas as pd
from exo_controller.datastream import Realtime_Datagenerator
from exo_controller.helpers import *
from exo_controller.MovementPrediction import MultiDimensionalDecisionTree
from exo_controller.emg_interface import EMG_Interface
from exo_controller.exo_controller import Exo_Control
from exo_controller.filter import MichaelFilter
from exo_controller.grid_arrangement import Grid_Arrangement
from exo_controller import ExtractImportantChannels
from exo_controller.DTW import *
from exo_controller import normalizations
from exo_controller.helpers import *
from exo_controller.spatial_filters import Filters
from exo_controller.ShallowConv2 import ShallowConvNetWithAttention
from exo_controller.muovipro import *
import time
import threading
import keyboard



class EMGProcessor:
    def __init__(
        self,
        patient_id,
        movements,
        grid_order,
        use_difference_heatmap,
        use_important_channels,
        use_local,
        output_on_exo,
        filter_output,
        time_for_each_movement_recording,
        load_trained_model,
        save_trained_model,
        use_spatial_filter,
        use_gauss_filter,
        use_bandpass_filter,
        use_mean_subtraction,
        use_recorded_data,
        window_size,
        scaling_method,
        only_record_data,
        use_control_stream,
        use_shallow_conv,
        use_virtual_hand_interface_for_coord_generation,
        epochs,
        use_dtw,
        use_muovi_pro

    ):
        self.patient_id = patient_id
        self.movements = movements
        self.grid_order = grid_order
        self.use_important_channels = use_important_channels
        self.use_local = use_local
        self.output_on_exo = output_on_exo
        self.filter_output = filter_output
        self.time_for_each_movement_recording = time_for_each_movement_recording
        self.load_trained_model = load_trained_model
        self.save_trained_model = save_trained_model
        self.use_mean_subtraction = use_mean_subtraction
        self.use_recorded_data = use_recorded_data
        self.window_size = window_size
        self.scaling_method = scaling_method
        self.use_gauss_filter = use_gauss_filter
        self.use_bandpass_filter = use_bandpass_filter
        self.use_spatial_filter = use_spatial_filter
        self.use_difference_heatmap = use_difference_heatmap
        self.only_record_data = only_record_data
        self.use_control_stream = use_control_stream
        self.use_shallow_conv = use_shallow_conv
        self.use_virtual_hand_interface_for_coord_generation = use_virtual_hand_interface_for_coord_generation
        self.retrain = False
        self.retrain_counter = 0
        self.epochs = epochs
        self.use_dtw = use_dtw
        self.use_muovi_pro = use_muovi_pro
        if use_muovi_pro:
            self.sampling_frequency = 2000
        else:
            self.sampling_frequency = 2048

        self.mean_rest = None
        self.model = None
        self.exo_controller = None
        if self.use_muovi_pro:
            self.emg_interface = Muoviprobe_Interface()
        else:
            self.emg_interface = EMG_Interface(grid_order=self.grid_order)

        self.grid_aranger = None
        self.gauss_filter = None
        self.normalizer = None

        self.initialize()

    def choose_finger_indexes(self):
        if "fist" in self.movements:
            self.finger_indexes = [0,2,3,4,5]
            return self.finger_indexes
        elif ("thumb" in self.movements) and ("index" in self.movements) and ("middle" in self.movements) and ("ring" in self.movements) and ("pinkie" in self.movements):
            self.finger_indexes = [0,2,3,4,5]
            return self.finger_indexes
        else:
            self.finger_indexes = []
            if "thumb" in self.movements:
                self.finger_indexes.append(0)
            if "index" in self.movements:
                self.finger_indexes.append(2)
            if "middle" in self.movements:
                self.finger_indexes.append(3)
            if "ring" in self.movements:
                self.finger_indexes.append(4)
            if "pinkie" in self.movements:
                self.finger_indexes.append(5)
            if "2pinch" in self.movements:
                if 0 not in self.finger_indexes:
                    self.finger_indexes.append(0)
                if 2 not in self.finger_indexes:
                    self.finger_indexes.append(2)
            if "3pinch" in self.movements:
                if 0 not in self.finger_indexes:
                    self.finger_indexes.append(0)
                if 2 not in self.finger_indexes:
                    self.finger_indexes.append(2)
                if 3 not in self.finger_indexes:
                    self.finger_indexes.append(3)
            self.finger_indexes.sort()
            return self.finger_indexes

    def initialize(self):
        self.finger_indexes = self.choose_finger_indexes()
        self.filter_local = MichaelFilter(num_fingers=len(self.finger_indexes))
        self.filter_time = MichaelFilter(num_fingers=len(self.finger_indexes))
        self.filter = Filters()
        print("using the following fingers: ", self.finger_indexes)
        self.grid_aranger = Grid_Arrangement(self.grid_order,use_muovi_pro=self.use_muovi_pro)
        self.grid_aranger.make_grid()

        if self.use_gauss_filter:
            self.gauss_filter = self.filter.create_gaussian_filter(size_filter=3)

    def run_video_data_generation(self):
        # Run video data generation
        patient = Realtime_Datagenerator(
            debug=False,
            patient_id=self.patient_id,
            sampling_frequency_emg=2048,
            recording_time=self.time_for_each_movement_recording,
            movements = self.movements.copy(),
            grid_order = self.grid_order,
            use_virtual_hand_interface_for_coord_generation = self.use_virtual_hand_interface_for_coord_generation,
            retrain = self.retrain,
            retrain_number = self.retrain_counter,
            finger_indexes = self.finger_indexes,
            use_muovi_pro = self.use_muovi_pro
        )
        patient.run_parallel()

    def remove_nan_values(self,data):
        """
        removes nan values from the data
        :param data:
        :return:
        """
        for i in data.keys():
            data[i] = np.array(data[i]).astype(np.float32)
            data[i][np.isnan(data[i])] = 0

        return data

    def load_data(self):
        # Load and preprocess data
        if (not self.use_recorded_data) and (not self.load_trained_model):
            self.run_video_data_generation()

        if self.retrain:
            self.run_video_data_generation()

            resulting_file = f"trainings_data/resulting_trainings_data/subject_{self.patient_id}/emg_data_retrain{self.retrain_counter}.pkl"
            emg_data = load_pickle_file(resulting_file)
            print("shape emg data: " ,emg_data["rest"].shape)
            ref_data = load_pickle_file(
                f"trainings_data/resulting_trainings_data/subject_{self.patient_id}/3d_data_retrain{self.retrain_counter}.pkl"
            )
            self.retrain_counter += 1
            print("shape ref data: ", ref_data["rest"].shape)


        else:
            resulting_file = f"trainings_data/resulting_trainings_data/subject_{self.patient_id}/emg_data.pkl"
            emg_data = load_pickle_file(resulting_file)
            print("shape emg data: ", emg_data["rest"].shape)
            ref_data = load_pickle_file(
                f"trainings_data/resulting_trainings_data/subject_{self.patient_id}/3d_data.pkl"
            )
            print("shape ref data: ", ref_data["rest"].shape)

        if self.use_dtw:
            for movement in ref_data.keys():
                if movement == "rest":
                    continue
                dtw_rms = make_rms_for_dtw(emg_data[movement])
                emg_data[movement], ref_data[movement] = align_signals_dtw(
                    dtw_rms, ref_data[movement],emg_data[movement]
                )

            print("shape emg rest data after dtw: ", emg_data["rest"].shape)
            print("shape ref rest data after dtw: ", ref_data["rest"].shape)
            print("shape emg thumb data after dtw: ", emg_data["thumb"].shape)
            print("shape ref thumb data after dtw: ", ref_data["thumb"].shape)


        return emg_data, ref_data

    def train_model(self, emg_data, ref_data):
        # Train or load the model
        if (self.load_trained_model is False) or (self.retrain is True):
            # Training a new model
            if self.retrain:
                model = MultiDimensionalDecisionTree(
                    important_channels=self.channels,
                    movements=self.movements.copy(),
                    emg=emg_data,
                    ref=ref_data,
                    patient_number=self.patient_id,
                    grid_order=self.grid_order,
                    mean_rest=self.mean_rest,
                    normalizer=self.normalizer,
                    use_gauss_filter=self.use_gauss_filter,
                    use_bandpass_filter=self.use_bandpass_filter,
                    filter_=self.filter,
                    use_difference_heatmap=self.use_difference_heatmap,
                    collected_with_virtual_hand=self.use_virtual_hand_interface_for_coord_generation,
                    retrain=True,
                    retrain_number = self.retrain_counter,
                    use_muovi_pro = self.use_muovi_pro,
                    use_spatial_filter=self.use_spatial_filter,
                )

            else:
                model = MultiDimensionalDecisionTree(
                    important_channels=self.channels,
                    movements=self.movements.copy(),
                    emg=emg_data,
                    ref=ref_data,
                    patient_number=self.patient_id,
                    grid_order=self.grid_order,
                    mean_rest=self.mean_rest,
                    normalizer=self.normalizer,
                    use_gauss_filter=self.use_gauss_filter,
                    use_bandpass_filter=self.use_bandpass_filter,
                    filter_=self.filter,
                    use_difference_heatmap=self.use_difference_heatmap,
                    collected_with_virtual_hand=self.use_virtual_hand_interface_for_coord_generation,
                    use_muovi_pro = self.use_muovi_pro,
                    use_spatial_filter=self.use_spatial_filter,
                )
            model.build_training_data(model.movements)
            model.save_trainings_data()
            self.num_previous_samples = model.num_previous_samples
            self.window_size_in_samples = model.window_size_in_samples

            self.best_time_tree = 2

            if self.use_shallow_conv:
                shallow_model = ShallowConvNetWithAttention(use_difference_heatmap=self.use_difference_heatmap ,best_time_tree=self.best_time_tree, grid_aranger=self.grid_aranger,number_of_grids=len(self.grid_order),use_mean=self.use_mean_subtraction,retrain=self.retrain,retrain_number= self.retrain_counter, finger_indexes=self.finger_indexes,use_muovi_pro=self.use_muovi_pro)
                if self.retrain:
                    shallow_model.load_model(cls=shallow_model, file_path=self.patient_id + "_shallow.pt")
                else:
                    shallow_model.apply(shallow_model._initialize_weights)

                train_loader,test_loader = shallow_model.load_trainings_data(self.patient_id)

                if self.retrain:
                    shallow_model.train_model(train_loader, epochs=50, learning_rate=0.000001)
                else:
                    shallow_model.train_model(train_loader, epochs=self.epochs) # 7
                    shallow_model.evaluate(test_loader)

            else:
                model.train()
                self.best_time_tree = model.evaluate(give_best_time_tree=True)


            if self.save_trained_model:
                if self.use_shallow_conv:
                    shallow_model.save_model(path=self.patient_id + "_shallow.pt")
                    print("Shallow model saved")
                else:
                    model.save_model(subject=self.patient_id)
                    print("Model saved")

            if self.use_shallow_conv:
                return shallow_model
            else:
                return model

        else:
            if self.use_shallow_conv:
                model = ShallowConvNetWithAttention(use_difference_heatmap=self.use_difference_heatmap ,best_time_tree=2, grid_aranger=self.grid_aranger,number_of_grids=len(self.grid_order),use_mean=self.use_mean_subtraction, finger_indexes=self.finger_indexes,use_muovi_pro=self.use_muovi_pro)
                model.load_model(cls = model,file_path=self.patient_id + "_shallow.pt")
                print("Shallow model loaded")
                self.best_time_tree = 3
                # Loading an existing model
            else:
                model = MultiDimensionalDecisionTree(
                    important_channels=self.channels,
                    movements=self.movements.copy(),
                    emg=None,
                    ref=None,
                    patient_number=self.patient_id,
                    grid_order=self.grid_order,
                    mean_rest=self.mean_rest,
                    normalizer=self.normalizer,
                    use_gauss_filter=self.use_gauss_filter,
                    use_bandpass_filter=self.use_bandpass_filter,
                    filter_=self.filter,
                    collected_with_virtual_hand=self.use_virtual_hand_interface_for_coord_generation,
                    use_muovi_pro=self.use_muovi_pro,
                    use_spatial_filter=self.use_spatial_filter,
                )
                model.load_model(subject=self.patient_id)
                self.best_time_tree = 1  # This might need to be adjusted based on how your model handles time trees
            return model

    def process_data(self, emg_data, ref_data):

        # Process data for model input
        for i in emg_data.keys():
            emg_data[i] = np.array(
                emg_data[i]#.transpose(1, 0, 2).reshape(len(self.grid_order)*64, -1)
            )  # reshape emg data such as it has the shape 320 x #samples for each movement

        copied_emg_data = emg_data.copy()
        for i in copied_emg_data.keys():
            copied_emg_data[i] =self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement(copied_emg_data[i])

        if self.use_important_channels:
            important_channels = extract_important_channels_realtime(
                self.movements.copy(), copied_emg_data, ref_data
            )
            self.channels = important_channels
            self.channels_row_shape = [
                self.grid_aranger.from_grid_position_to_row_position(ch[0],ch[1]) for ch in important_channels
            ]

        else:
            if self.use_muovi_pro:
                self.channels = range(32)
            else:
                self.channels = range(len(self.grid_order) * 64)

        self.normalizer = normalizations.Normalization(
            method=self.scaling_method,
            grid_order=self.grid_order,
            important_channels=self.channels,
            frame_duration=self.window_size,
            use_spatial_filter=self.use_spatial_filter,
            use_muovi_pro=self.use_muovi_pro
        )

        #shape should be 320 x #samples
        #ref_data = resample_reference_data(ref_data, emg_data)
        ref_data = self.remove_nan_values(ref_data)
        emg_data = self.remove_nan_values(emg_data)
        print("length of emg data: ", len(emg_data["rest"][0]))
        print("length of ref data: ", ref_data["rest"].shape[0])

        for i in emg_data.keys():
            emg_data[i] = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement(emg_data[i])



        #shape should be grid
        if self.use_mean_subtraction:
            channel_extractor = ExtractImportantChannels.ChannelExtraction("rest", emg_data, ref_data,use_gaussian_filter=self.use_gauss_filter,use_muovi_pro=self.use_muovi_pro,use_spatial_filter=self.use_spatial_filter)
            self.mean_rest, _, _ = channel_extractor.get_heatmaps()
            self.normalizer.set_mean(mean=self.mean_rest)

        # add gaussian noise to the ground truth data so that the model can learn to deal with noise
        for movement in self.movements:
            # Generate Gaussian noise for each column
            for finger in range(len(self.finger_indexes)):
                std_i = np.divide(np.std(ref_data[movement][:, finger]), 10)
                noise_i = np.random.normal(0, std_i, ref_data[movement].shape[0])
                ref_data[movement][:, finger] = np.add(ref_data[movement][:, finger], noise_i)

        # Calculate normalization values
        self.normalizer.get_all_emg_data(
            path_to_data=f"trainings_data/resulting_trainings_data/subject_{self.patient_id}/emg_data.pkl",
            movements=self.movements.copy(),
        )
        self.normalizer.calculate_normalization_values()


    def format_for_exo(self, results,control_results=None):
        # exo receives 18 values, the first 8 as the values of the predicted hand and the other 8 as the values of the control hand
        if self.use_control_stream:
            res = [0]*18
            count = 0
            for i in self.finger_indexes:
                res[i] = round(results[count], 3)
                res[i+9] = round(control_results[count], 3)
                count += 1
            return res
        else:
            res = [0]*9
            count = 0
            for i in self.finger_indexes:
                res[i] = round(results[count], 3)
                count += 1
            return res


    def output_results(self, results,control_results=None):

        for i in range(len(self.finger_indexes)):
            if results[i] > 1:
                results[i] = 1
            if results[i] < 0:
                results[i] = 0

        # Output the results either to the exoskeleton or console
        if self.output_on_exo:
            # Logic to send commands to the exoskeleton
            # Assuming 'results' is in the format expected by the exoskeleton
            # You might need to format 'results' accordingly
            formatted_results = self.format_for_exo(results,control_results)
            self.exo_controller.move_exo(formatted_results)
        else:
            # Print the prediction results to the console
            print("Prediction: ", results)

    def run_prediction_loop_recorded_data(self, model):
        self.exo_controller = Exo_Control()
        self.exo_controller.initialize_all()
        # Main prediction loop
        if self.use_muovi_pro:
            max_chunk_number = np.ceil(
                max(self.num_previous_samples) / 18
            )  # calculat
        else:
            max_chunk_number = np.ceil(
                max(self.num_previous_samples) / 64
            )  # calculate number of how many chunks we have to store till we delete old

        emg_data = load_pickle_file(self.use_recorded_data + "emg_data.pkl")
        for i in emg_data.keys():
            emg_data[i] = np.array(
                emg_data[i]#.transpose(1, 0, 2).reshape(len(self.grid_order) * 64, -1)
            )
            if self.use_important_channels:
                for channel in range(emg_data[i].shape[0]):
                    if channel not in self.channels_row_shape:
                        emg_data[i][channel,:] = 0
        emg_data = self.remove_nan_values(emg_data)

        ref_data = load_pickle_file(self.use_recorded_data + "3d_data.pkl")
        ref_data = self.remove_nan_values(ref_data)
        #ref_data = resample_reference_data(ref_data, emg_data)
        if self.use_dtw:
            for movement in ref_data.keys():
                if movement == "rest":
                    continue
                dtw_rms = make_rms_for_dtw(emg_data[movement])
                emg_data[movement], ref_data[movement] = align_signals_dtw(
                    dtw_rms, ref_data[movement],emg_data[movement]
                )

            print("shape emg rest data after dtw: ", emg_data["rest"].shape)
            print("shape ref rest data after dtw: ", ref_data["rest"].shape)
            print("shape emg thumb data after dtw: ", emg_data["thumb"].shape)
            print("shape ref thumb data after dtw: ", ref_data["thumb"].shape)

        for i in emg_data.keys():
            emg_data[i] = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement(
                emg_data[i]
            )


        if self.use_bandpass_filter:
            for i in emg_data.keys():
                emg_data[i] = self.filter.bandpass_filter_emg_data(emg_data[i], fs=self.sampling_frequency)


        for movement in ref_data.keys():
            buffer_pred = []
            buffer_control = []
            if movement in self.movements:

                # if movement != "rest":
                #     ref_data_for_this_movement = normalize_2D_array(ref_data[movement], axis=0)
                # else:
                #     ref_data_for_this_movement = ref_data[movement] * 0

                # if movement == "2pinch":
                #
                #     ref_data_for_this_movement[:, 0] = np.multiply(ref_data_for_this_movement[:, 0], 0.45)
                #     ref_data_for_this_movement[:, 1] = np.multiply(ref_data_for_this_movement[:, 1], 0.6)

                # if movement == "thumb":
                #     ref_data_for_this_movement = np.hstack((ref_data_for_this_movement, np.zeros((ref_data[movement].shape[0], 1))))
                #
                # if movement == "index":
                #     ref_data_for_this_movement = np.hstack((np.zeros((ref_data[movement].shape[0], 1)), ref_data_for_this_movement))
                ref_data_for_this_movement = ref_data[movement]

                emg_buffer = []
                # ref_data[movement] = normalize_2D_array(ref_data[movement], axis=0)
                print("movement: ", movement, file=sys.stderr)
                if self.use_muovi_pro:
                    skipping_samples = 18
                else:
                    skipping_samples = 64
                for sample in tqdm.tqdm(range(0,emg_data[movement].shape[2]- int((self.window_size / 1000) * self.sampling_frequency), skipping_samples)):
                    if (sample <= ref_data[movement].shape[0]) and (sample - max(self.num_previous_samples) >= 0):
                        time_start = time.time()
                        chunk = emg_data[movement][:, :, sample : sample + skipping_samples]
                        emg_buffer.append(chunk)
                        if (
                            len(emg_buffer) > max_chunk_number
                        ):  # check if now too many sampels are in the buffer and i can delete old one
                            emg_buffer.pop(0)
                        data = np.concatenate(emg_buffer, axis=-1)
                        if ((data.shape[2] - self.window_size_in_samples) < 0):
                            emg_to_use = data[:,:,:]
                        else:
                            emg_to_use = data[:,:,
                                         (data.shape[2] - 1) - self.window_size_in_samples: -1
                                         ]
                        if self.use_difference_heatmap:
                            # data for difference heatmap
                            if ((data.shape[2] - (self.window_size_in_samples * 2.5)) < 0):
                                emg_to_use_difference = data[:,:,:]
                            else:
                                emg_to_use_difference = data[:,:,
                                                        (data.shape[2] - 1) - self.window_size_in_samples: -1
                                                        ]

                        if self.use_spatial_filter:
                            emg_to_use = self.filter.spatial_filtering(emg_to_use, "IR")
                            if self.use_difference_heatmap:
                                emg_to_use_difference = self.filter.spatial_filtering(emg_to_use_difference, "IR")

                        heatmap_local = calculate_local_heatmap_realtime(
                            emg_to_use
                        )
                        if self.use_difference_heatmap:
                            previous_heatmap = calculate_local_heatmap_realtime(
                                emg_to_use_difference
                            )

                        if self.use_mean_subtraction:
                            heatmap_local = np.subtract(heatmap_local, self.mean_rest)
                            if self.use_difference_heatmap:
                                previous_heatmap = np.subtract(heatmap_local, self.mean_rest)

                        heatmap_local = self.normalizer.normalize_chunk(heatmap_local)
                        if self.use_difference_heatmap:
                            previous_heatmap = self.normalizer.normalize_chunk(previous_heatmap)


                        if self.use_gauss_filter:
                            heatmap_local = self.filter.apply_gaussian_filter(
                                heatmap_local, self.gauss_filter
                            )
                            if self.use_difference_heatmap:
                                previous_heatmap = self.filter.apply_gaussian_filter(
                                    previous_heatmap, self.gauss_filter
                                )
                        if self.use_difference_heatmap:
                            difference_heatmap = np.subtract(heatmap_local, previous_heatmap)
                            if not self.use_shallow_conv:
                                difference_heatmap = np.squeeze(self.grid_aranger.transfer_grid_arangement_into_320(
                                    np.reshape(difference_heatmap, (difference_heatmap.shape[0], difference_heatmap.shape[1], 1))))
                        if not self.use_shallow_conv:
                            heatmap_local = np.squeeze(self.grid_aranger.transfer_grid_arangement_into_320(
                                np.reshape(heatmap_local, (heatmap_local.shape[0], heatmap_local.shape[1], 1))))

                        if self.use_shallow_conv:
                            if not self.use_difference_heatmap:
                                res_local = model.predict(heatmap_local)
                                if self.filter_output:
                                    res_local = self.filter_local.filter(
                                        np.array(res_local[0])
                                    )  # filter the predcition with my filter from my Bachelor thesis

                                else:
                                    res_local = np.array(res_local[0])

                        else:
                            res_local = model.trees[0].predict(
                                [heatmap_local]
                            )  # result has shape 1,2

                            if self.filter_output:
                                res_local = self.filter_local.filter(
                                    np.array(res_local[0])
                                )  # filter the predcition with my filter from my Bachelor thesis

                            else:
                                res_local = np.array(res_local[0])

                        if self.use_difference_heatmap:
                            if np.isnan(difference_heatmap).any():
                                res_time = np.array([-1]*len(self.finger_indexes))
                            else:
                                if self.use_shallow_conv:
                                    res_time = model.predict(heatmap_local,difference_heatmap)
                                else:
                                    res_time = model.trees[self.best_time_tree].predict(
                                        [difference_heatmap]
                                    )

                                if self.filter_output:
                                    res_time = self.filter_time.filter(
                                        np.array(res_time[0])
                                    )  # fileter the predcition with my filter from my Bachelor thesis
                                else:
                                    res_time = np.array(res_time[0])

                        if not self.use_virtual_hand_interface_for_coord_generation:
                            control_ref_data = ref_data_for_this_movement[sample]
                        else:
                            control_ref_data = ref_data[movement][sample]

                        if self.use_local:
                            if self.use_control_stream:
                                buffer_pred.append(res_local)
                                buffer_control.append(control_ref_data)
                                self.output_results(res_local,control_ref_data)
                            else:
                                self.output_results(res_local)
                        else:
                            if self.use_control_stream:
                                self.output_results(res_time,control_ref_data)
                            else:
                                self.output_results(res_time)
                        time_end = time.time()
                        if time_end - time_start < (skipping_samples/self.sampling_frequency):
                            time.sleep((skipping_samples/self.sampling_frequency) - (time_end - time_start))

            # plt.figure()
            # for finger in range(len(self.finger_indexes)):
            #     plt.plot(np.array(buffer_pred)[:,finger],label="pred " + str(finger))
            #     plt.plot(np.add(np.array(buffer_control)[:,finger],finger),label="ref " + str(finger))
            # plt.legend()
            # plt.show()

    def run_prediction_loop(self, model):

        self.emg_interface.initialize_all()
        self.exo_controller = Exo_Control()
        self.exo_controller.initialize_all()
        if self.use_muovi_pro:
            skip_samples = 18
        else:
            skip_samples = 64

        if not self.load_trained_model:
            max_chunk_number = np.ceil(
                max(self.num_previous_samples) / skip_samples
            )  # calculate number of how many chunks we have to store till we delete old
        else:
            max_chunk_number = 15
            self.window_size_in_samples = int((self.window_size / 1000) * self.sampling_frequency)
        emg_buffer = []


        self.retrain = False
        retrain_input_thread = threading.Thread(target=self.check_input)
        retrain_input_thread.start()

        while True:
            if self.retrain:
                break
            # try:
            # clear all data that came into the buffer since last time (do not want to process old data)
            self.emg_interface.clear_socket_buffer()
            chunk = self.emg_interface.get_EMG_chunk()
            emg_buffer.append(chunk)
            if (
                    len(emg_buffer) > max_chunk_number
            ):  # check if now too many sampels are in the buffer and i can delete old one
                emg_buffer.pop(0)
            data = np.concatenate(emg_buffer, axis=-1)
            if self.use_bandpass_filter:
                data = self.filter.bandpass_filter_emg_data(data, fs=self.sampling_frequency)
            data = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement(data)

            if ((data.shape[2] - self.window_size_in_samples) < 0):
                emg_to_use = data[:, :, :]
            else:
                emg_to_use = data[:, :,
                             (data.shape[2] - 1) - self.window_size_in_samples: -1
                             ]
            if self.use_difference_heatmap:
                # data for difference heatmap
                if ((data.shape[2] - (self.window_size_in_samples * 2.5)) < 0):
                    emg_to_use_difference = data[:, :, :]
                else:
                    emg_to_use_difference = data[:, :,
                                            (data.shape[2] - 1) - int(self.window_size_in_samples*2.5): -1
                                            ]

            if self.use_spatial_filter:
                emg_to_use = self.filter.spatial_filtering(emg_to_use, "IR")
                if self.use_difference_heatmap:
                    emg_to_use_difference = self.filter.spatial_filtering(emg_to_use_difference, "IR")

            heatmap_local = calculate_local_heatmap_realtime(
                emg_to_use
            )
            if self.use_difference_heatmap:
                previous_heatmap = calculate_local_heatmap_realtime(
                    emg_to_use_difference
                )

            if self.use_mean_subtraction:
                heatmap_local = np.subtract(heatmap_local, self.mean_rest)
                if self.use_difference_heatmap:
                    previous_heatmap = np.subtract(heatmap_local, self.mean_rest)

            heatmap_local = self.normalizer.normalize_chunk(heatmap_local)
            if self.use_difference_heatmap:
                previous_heatmap = self.normalizer.normalize_chunk(previous_heatmap)

            if self.use_gauss_filter:
                heatmap_local = self.filter.apply_gaussian_filter(
                    heatmap_local, self.gauss_filter
                )
                if self.use_difference_heatmap:
                    previous_heatmap = self.filter.apply_gaussian_filter(
                        previous_heatmap, self.gauss_filter
                    )
            if self.use_difference_heatmap:
                difference_heatmap = previous_heatmap
                if not self.use_shallow_conv:
                    difference_heatmap = np.squeeze(self.grid_aranger.transfer_grid_arangement_into_320(
                        np.reshape(difference_heatmap, (difference_heatmap.shape[0], difference_heatmap.shape[1], 1))))
            if not self.use_shallow_conv:
                heatmap_local = np.squeeze(self.grid_aranger.transfer_grid_arangement_into_320(
                    np.reshape(heatmap_local, (heatmap_local.shape[0], heatmap_local.shape[1], 1))))

            if self.use_shallow_conv:
                if not self.use_difference_heatmap:
                    res_local = model.predict(heatmap_local)
                    if self.filter_output:
                        res_local = self.filter_local.filter(
                            np.array(res_local[0])
                        )  # filter the predcition with my filter from my Bachelor thesis

                    else:
                        res_local = np.array(res_local[0])

            else:
                res_local = model.trees[0].predict(
                    [heatmap_local]
                )  # result has shape 1,2

                if self.filter_output:
                    res_local = self.filter_local.filter(
                        np.array(res_local[0])
                    )  # filter the predcition with my filter from my Bachelor thesis

                else:
                    res_local = np.array(res_local[0])

            if self.use_difference_heatmap:
                if np.isnan(difference_heatmap).any():
                    res_time = np.array([-1]*len(self.finger_indexes))
                else:
                    if self.use_shallow_conv:
                        res_time = model.predict(heatmap_local, difference_heatmap)
                    else:
                        res_time = model.trees[self.best_time_tree].predict(
                            [difference_heatmap]
                        )

                    if self.filter_output:
                        res_time = self.filter_time.filter(
                            np.array(res_time[0])
                        )  # fileter the predcition with my filter from my Bachelor thesis
                    else:
                        res_time = np.array(res_time[0])


            if self.use_local:
                self.output_results(res_local)
            else:
                self.output_results(res_time)
        if self.retrain:
            self.retrain_model()

    def check_input(self):
        while True:
            user_input = input("Press r for retrain!! ")
            if user_input == "r":  # replace 'certain_input' with your specific condition
                print("Retraining will be started")
                # Do something when the specific input is received
                self.retrain = True
                break


    def retrain_model(self):
        self.load_trained_model = True
        self.time_for_each_movement_recording =10
        self.emg_interface.close_connection()
        self.exo_controller.close_connection()
        self.run()


    def run(self):
        # Main loop for predictions
        emg_data, ref_data = self.load_data()
        if self.only_record_data:
            return
        self.process_data(emg_data, ref_data)

        model = self.train_model(emg_data, ref_data)
        # if self.use_recorded_data:
        #     self.run_prediction_loop_recorded_data(model)
        # else:
        #     self.run_prediction_loop(model)
        self.run_prediction_loop(model)


if __name__ == "__main__":
    # "Min_Max_Scaling_over_whole_data" = min max scaling with max/min is choosen individually for every channel
    # "Robust_all_channels" = robust scaling with q1,q2,median is choosen over all channels
    # "Robust_Scaling"  = robust scaling with q1,q2,median is choosen individually for every channel
    # "Min_Max_Scaling_all_channels" = min max scaling with max/min is choosen over all channels

    emg_processor = EMGProcessor(
        patient_id="Test",
        movements=[
            "rest",
            "thumb",
            "index",
            # "2pinch",
            # "3pinch",
            "middle",
            "ring",
            "pinkie",
            # "fist",
        ],
        grid_order=[1,2,3,4,5],
        use_difference_heatmap=False,
        use_important_channels=False,
        use_local=True,
        output_on_exo=True,
        filter_output=True,
        time_for_each_movement_recording=30,
        load_trained_model=False,
        save_trained_model=True,
        use_spatial_filter=False,
        use_mean_subtraction=True,
        use_bandpass_filter=False,
        use_gauss_filter=True,
        use_recorded_data=r"trainings_data/resulting_trainings_data/subject_Test/",  # False
        window_size=150,
        scaling_method="Min_Max_Scaling_over_whole_data",
        only_record_data=False,
        use_control_stream=False,
        use_shallow_conv=True,
        use_virtual_hand_interface_for_coord_generation = True,
        epochs=200,
        use_dtw=False,
        use_muovi_pro=True,

    )
    emg_processor.run()




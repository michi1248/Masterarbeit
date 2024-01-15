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
from exo_controller import normalizations
from exo_controller.helpers import *
from exo_controller.spatial_filters import Filters
from exo_controller.ShallowConv2 import ShallowConvNetWithAttention
import time
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
        use_virtual_hand_interface_for_coord_generation
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

        self.mean_rest = None
        self.model = None
        self.exo_controller = None
        self.emg_interface = EMG_Interface(grid_order=self.grid_order)
        self.filter_local = MichaelFilter()
        self.filter_time = MichaelFilter()
        self.filter = Filters()
        self.grid_aranger = None
        self.gauss_filter = None
        self.normalizer = None

        self.initialize()

    def initialize(self):

        self.grid_aranger = Grid_Arrangement(self.grid_order)
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
            use_virtual_hand_interface_for_coord_generation = self.use_virtual_hand_interface_for_coord_generation
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
        if not self.use_recorded_data:
            self.run_video_data_generation()

        resulting_file = f"trainings_data/resulting_trainings_data/subject_{self.patient_id}/emg_data.pkl"
        emg_data = load_pickle_file(resulting_file)
        print("shape emg data: " ,emg_data["rest"].shape)
        ref_data = load_pickle_file(
            f"trainings_data/resulting_trainings_data/subject_{self.patient_id}/3d_data.pkl"
        )
        print("shape ref data: ", ref_data["rest"].shape)
        ref_data["rest"] = ref_data["rest"]*0
        return emg_data, ref_data

    def train_model(self, emg_data, ref_data):
        # Train or load the model
        if not self.load_trained_model:
            # Training a new model
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
                collected_with_virtual_hand=self.use_virtual_hand_interface_for_coord_generation
            )
            model.build_training_data(model.movements)
            model.save_trainings_data()
            self.num_previous_samples = model.num_previous_samples
            self.window_size_in_samples = model.window_size_in_samples

            self.best_time_tree = 2

            if self.use_shallow_conv:
                shallow_model = ShallowConvNetWithAttention(use_difference_heatmap=self.use_difference_heatmap ,best_time_tree=self.best_time_tree, grid_aranger=self.grid_aranger,number_of_grids=len(self.grid_order))
                shallow_model.apply(shallow_model._initialize_weights)
                train_loader,test_loader = shallow_model.load_trainings_data(self.patient_id)
                shallow_model.train_model(train_loader, epochs=500) # 7
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
                model = ShallowConvNetWithAttention(number_of_grids=len(self.grid_order))
                model.load_model(path=self.patient_id + "_shallow.pt")
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
                    collected_with_virtual_hand=self.use_virtual_hand_interface_for_coord_generation
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


        if self.use_important_channels:
            important_channels = extract_important_channels_realtime(
                self.movements.copy(), emg_data, ref_data
            )
            self.channels = [
                self.grid_aranger.from_grid_position_to_row_position(ch) for ch in important_channels
            ]
        else:
            self.channels = range(len(self.grid_order) * 64)

        self.normalizer = normalizations.Normalization(
            method=self.scaling_method,
            grid_order=self.grid_order,
            important_channels=self.channels,
            frame_duration=self.window_size,
        )

        #shape should be 320 x #samples
        ref_data = resample_reference_data(ref_data, emg_data)
        ref_data = self.remove_nan_values(ref_data)
        emg_data = self.remove_nan_values(emg_data)
        print("length of emg data: ", len(emg_data["rest"][0]))
        print("length of ref data: ", ref_data["rest"].shape[0])

        for i in emg_data.keys():
            emg_data[i] = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement(emg_data[i])



        #shape should be grid
        if self.use_mean_subtraction:
            channel_extractor = ExtractImportantChannels.ChannelExtraction("rest", emg_data, ref_data)
            self.mean_rest, _, _ = channel_extractor.get_heatmaps()
            self.normalizer.set_mean(mean=self.mean_rest)

        # Calculate normalization values
        self.normalizer.get_all_emg_data(
            path_to_data=f"trainings_data/resulting_trainings_data/subject_{self.patient_id}/emg_data.pkl",
            movements=self.movements.copy(),
        )
        self.normalizer.calculate_normalization_values()


    def format_for_exo(self, results,control_results=None):
        # Format the results in a way that is compatible with the exoskeleton
        # This is a placeholder - you'll need to replace it with actual logic
        # Example: [round(result, 3) for result in results]
        if self.use_control_stream:
            res = [
                round(results[0], 3),
                0,
                round(results[1], 3),
                0,
                0,
                0,
                0,
                0,
                0,
                round(control_results[0], 3),
                0,
                round(control_results[1], 3),
                0,
                0,
                0,
                0,
                0,
                0,
            ]
            return res
        else:
            res = [
                round(results[0], 3),
                0,
                round(results[1], 3),
                0,
                0,
                0,
                0,
                0,
                0,
            ]
            return res


    def output_results(self, results,control_results=None):

        for i in range(2):
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
        max_chunk_number = np.ceil(
            max(self.num_previous_samples) / 64
        )  # calculate number of how many chunks we have to store till we delete old

        emg_data = load_pickle_file(self.use_recorded_data + "emg_data.pkl")
        for i in emg_data.keys():
            emg_data[i] = np.array(
                emg_data[i]#.transpose(1, 0, 2).reshape(len(self.grid_order) * 64, -1)
            )
        emg_data = self.remove_nan_values(emg_data)

        ref_data = load_pickle_file(self.use_recorded_data + "3d_data.pkl")
        ref_data = self.remove_nan_values(ref_data)
        ref_data = resample_reference_data(ref_data, emg_data)



        for i in emg_data.keys():
            emg_data[i] = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement(
                emg_data[i]
            )


        if self.use_bandpass_filter:
            for i in emg_data.keys():
                emg_data[i] = self.filter.bandpass_filter_emg_data(emg_data[i], fs=2048)


        if self.use_spatial_filter:
            for i in emg_data.keys():
                emg_data[i] = self.filter.spatial_filtering(emg_data[i], "NDD")


        for movement in ref_data.keys():
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
                for sample in tqdm.tqdm(range(0,emg_data[movement].shape[2], 64)):
                    time_start = time.time()
                    chunk = emg_data[movement][:, :, sample : sample + 64]
                    emg_buffer.append(chunk)
                    if (
                        len(emg_buffer) > max_chunk_number
                    ):  # check if now too many sampels are in the buffer and i can delete old one
                        emg_buffer.pop(0)
                    data = np.concatenate(emg_buffer, axis=-1)
                    heatmap_local = calculate_local_heatmap_realtime(
                        data, self.window_size_in_samples
                    )
                    if self.use_difference_heatmap:
                        previous_heatmap = calculate_difference_heatmap_realtime(
                            data,
                            data.shape[2]
                            - self.num_previous_samples[self.best_time_tree - 1],
                            self.window_size_in_samples,
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
                            res_time = np.array([-1, -1])
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
                            self.output_results(res_local,control_ref_data)
                        else:
                            self.output_results(res_local)
                    else:
                        if self.use_control_stream:
                            self.output_results(res_time,control_ref_data)
                        else:
                            self.output_results(res_time)
                    time_end = time.time()
                    if time_end - time_start < (64/2048):
                        time.sleep((64/2048) - (time_end - time_start))


    def run_prediction_loop(self, model):

        self.emg_interface.initialize_all()
        max_chunk_number = np.ceil(
            max(self.num_previous_samples) / 64
        )  # calculate number of how many chunks we have to store till we delete old
        emg_buffer = []

        while True:
            # try:
            chunk = self.emg_interface.get_EMG_chunk()
            emg_buffer.append(chunk)
            if (
                    len(emg_buffer) > max_chunk_number
            ):  # check if now too many sampels are in the buffer and i can delete old one
                emg_buffer.pop(0)
            data = np.concatenate(emg_buffer, axis=-1)


            data = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement(data)

            if self.use_bandpass_filter:
                data = self.filter.bandpass_filter_emg_data(data, fs=2048)

            if self.use_spatial_filter:
                data = self.filter.spatial_filtering(data, "IB2")

            heatmap_local = calculate_local_heatmap_realtime(
                data, self.window_size_in_samples
            )
            if self.use_difference_heatmap:
                previous_heatmap = calculate_difference_heatmap_realtime(
                    data,
                    data.shape[2]
                    - self.num_previous_samples[self.best_time_tree - 1],
                    self.window_size_in_samples,
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
            # TODO hier muss noch die heatmap in die richtige form gebracht werden
            #TODO hier müsste eigentlich grid shape dann gemacht werden
            if not self.use_shallow_conv:
                heatmap_local = np.squeeze(self.grid_aranger.transfer_grid_arangement_into_320(
                    np.reshape(heatmap_local, (heatmap_local.shape[0], heatmap_local.shape[1], 1))))

            if self.use_shallow_conv:
                res_local = model.predict(heatmap_local)

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
                    res_time = np.array([-1, -1])
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

            if self.use_local:
                print(res_local)
                self.output_results(res_local)
            else:
                self.output_results(res_time)

    def run(self):
        # Main loop for predictions

        emg_data, ref_data = self.load_data()
        if self.only_record_data:
            return

        self.process_data(emg_data, ref_data)
        model = self.train_model(emg_data, ref_data)
        if self.use_recorded_data:
            self.run_prediction_loop_recorded_data(model)
        else:
            self.run_prediction_loop(model)


if __name__ == "__main__":
    # "Min_Max_Scaling_over_whole_data" = min max scaling with max/min is choosen individually for every channel
    # "Robust_all_channels" = robust scaling with q1,q2,median is choosen over all channels
    # "Robust_Scaling"  = robust scaling with q1,q2,median is choosen individually for every channel
    # "Min_Max_Scaling_all_channels" = min max scaling with max/min is choosen over all channels

    emg_processor = EMGProcessor(
        patient_id="Michi_11_01_2024_remapped2",
        movements=[
            "rest",
            "thumb",
            "index",
            "2pinch",
        ],
        grid_order=[1,2],
        use_difference_heatmap=False,
        use_important_channels=False,
        use_local=True,
        output_on_exo=True,
        filter_output=True,
        time_for_each_movement_recording=25,
        load_trained_model=False,
        save_trained_model=True,
        use_spatial_filter=False,
        use_mean_subtraction=False,
        use_bandpass_filter=False,
        use_gauss_filter=True,
        use_recorded_data=r"trainings_data/resulting_trainings_data/subject_Michi_11_01_2024_remapped2/",  # False
        window_size=150,
        scaling_method="no_scaling",
        only_record_data=False,
        use_control_stream=True,
        use_shallow_conv=True,
        use_virtual_hand_interface_for_coord_generation = True
    )
    emg_processor.run()




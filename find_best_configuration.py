import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

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
#from exo_controller.ShallowConv import ShallowConvNetWithAttention
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
        use_virtual_hand_interface_for_coord_generation,
        epochs,
        split_index_if_same_dataset = None
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
        self.epochs = epochs

        self.split_index_if_same_dataset = split_index_if_same_dataset

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

    def train_model(self, emg_data, ref_data,already_build=False):

        # Training a new model
        if not already_build:
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
            print("num_previous_samples: ", self.num_previous_samples)




        self.best_time_tree = 2
        if not self.use_shallow_conv:
            model.train()
            model.evaluate()
        else:
            shallow_model = ShallowConvNetWithAttention(use_difference_heatmap=self.use_difference_heatmap ,best_time_tree=self.best_time_tree, grid_aranger=self.grid_aranger,number_of_grids=len(self.grid_order),use_mean=self.use_mean_subtraction)
            shallow_model.apply(shallow_model._initialize_weights)
            train_loader,test_loader = shallow_model.load_trainings_data(self.patient_id)
            shallow_model.train_model(train_loader, epochs=self.epochs)
            shallow_model.evaluate(test_loader)
            self.train_loader = train_loader
        if self.save_trained_model:
            if not self.use_shallow_conv:
                model.save_model(subject=self.patient_id )
                print("Model saved")
            else:
                shallow_model.save_model(path=self.patient_id + "_shallow.pt")
            print("Shallow model saved")

        if self.use_shallow_conv:
            return shallow_model
        else:
            return model



    def process_data(self, emg_data, ref_data):

        # Process data for model input
        # for i in emg_data.keys():
        #     emg_data[i] = np.array(
        #         emg_data[i].transpose(1, 0, 2).reshape(len(self.grid_order)*64, -1)
        #     )  # reshape emg data such as it has the shape 320 x #samples for each movement

        copied_emg_data = emg_data.copy()
        for i in copied_emg_data.keys():
            copied_emg_data[i] = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement(copied_emg_data[i])

        if self.use_important_channels:
            important_channels = extract_important_channels_realtime(
                self.movements.copy(), copied_emg_data, ref_data
            )
            self.channels = important_channels
            self.channels_row_shape = [
                self.grid_aranger.from_grid_position_to_row_position(ch[0], ch[1]) for ch in important_channels
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
        predictions = []
        ground_truth = []
        # Main prediction loop
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
                ref_data_for_this_movement = ref_data[movement]
                emg_buffer = []
                # ref_data[movement] = normalize_2D_array(ref_data[movement], axis=0)
                print("movement: ", movement, file=sys.stderr)
                for sample in tqdm.tqdm(range(0,emg_data[movement].shape[2]- int((self.window_size / 1000) * 2048) + 1, 64)):
                    if (sample <= ref_data[movement].shape[0]) and (sample - max(self.num_previous_samples) >= 0):
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
                            previous_heatmap = calculate_local_heatmap_realtime(
                            data, int(self.window_size_in_samples*3.5)
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
                            difference_heatmap =  previous_heatmap
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
                                predictions.append(res_local)
                                ground_truth.append(control_ref_data)
                            else:
                                self.output_results(res_local)

                        else:
                            if self.use_control_stream:
                                predictions.append(res_time)
                                ground_truth.append(control_ref_data)
                            else:
                                self.output_results(res_time)
        return predictions,ground_truth


    def run(self,already_build=False):
        # Main loop for predictions

        if not already_build:
            emg_data, ref_data = self.load_data()
            if self.only_record_data:
                return
            self.process_data(emg_data, ref_data)
            model = self.train_model(emg_data, ref_data,already_build=already_build)
        else:
            model = self.train_model(None, None, already_build=already_build)

        results,ground_truth = self.run_prediction_loop_recorded_data(model)
        #print("difference: ", np.subtract(np.array([target for _, target in self.train_loader]).squeeze()[0],ground_truth))
        avg_loss,mse_loss = model.evaluate_best(predictions=results,ground_truth=ground_truth)
        return avg_loss,mse_loss


if __name__ == "__main__":
    # "Min_Max_Scaling_over_whole_data" = min max scaling with max/min is choosen individually for every channel
    # "Robust_all_channels" = robust scaling with q1,q2,median is choosen over all channels
    # "Robust_Scaling"  = robust scaling with q1,q2,median is choosen individually for every channel
    # "Min_Max_Scaling_all_channels" = min max scaling with max/min is choosen over all channels

    count = 0
    num_previous_samples = None
    window_size_in_samples = None
    normalizer_mean = None
    mean_rest = None
    grid_arranger = None

    important_channels_mean = None
    channels_row_shape_mean = None
    important_channels_no_mean = None
    channels_row_shape_no_mean = None

    normalizer_no_mean = None

    split_index_if_same_dataset = 0.8

    use_shallow_conv = True

    for method in  ["Min_Max_Scaling_over_whole_data"]:#["Min_Max_Scaling_all_channels","Min_Max_Scaling_over_whole_data","Robust_Scaling","Robust_all_channels","no_scaling"]:
        evaluation_results_mean_sub = []
        evaluation_results_no_mean_sub = []
        mse_evaluation_results_mean_sub = []
        mse_evaluation_results_no_mean_sub = []

        best_r2_mean = None
        best_r2_no_mean = None
        best_mse_mean = None
        best_mse_no_mean = None


        for epochs in [1,10,20,25,35,50,100,125,150,175,200,250,500,1000,1500,2000,2500,3000,4000]:#[1,5,10,15,20,25,30,40,50,60,70,100,250,500,1000,1500,2000,2500]:
            for use_mean_sub in [True, False]:  # [True,False]
                if (count > 0) and use_shallow_conv is False:
                    continue
                print("epochs: ", epochs)
                print("use_mean_sub: ", use_mean_sub)
                emg_processor = EMGProcessor(
                    patient_id="Michi_11_01_2024_normal2",
                    movements=[
                        "rest",
                        "thumb",
                        "index",
                        "2pinch",
                    ],
                    grid_order=[1,2],
                    use_difference_heatmap=False,
                    use_important_channels=True,
                    use_local=True,
                    output_on_exo=True,
                    filter_output=False,
                    time_for_each_movement_recording=25,
                    load_trained_model=False,
                    save_trained_model=True,
                    use_spatial_filter=False,
                    use_mean_subtraction=use_mean_sub,
                    use_bandpass_filter=False,
                    use_gauss_filter=True,
                    use_recorded_data=r"trainings_data/resulting_trainings_data/subject_Michi_11_01_2024_normal3/",  # False
                    window_size=150,
                    scaling_method=method,
                    only_record_data=False,
                    use_control_stream=True,
                    use_shallow_conv=use_shallow_conv,
                    #set this to false if not recorded with virtual hand interface
                    use_virtual_hand_interface_for_coord_generation = True,
                    epochs = epochs,
                    split_index_if_same_dataset = split_index_if_same_dataset

                )
                if count <= 1:
                    avg_loss, mse_loss = emg_processor.run(already_build=False)
                else:
                    emg_processor.num_previous_samples = num_previous_samples
                    emg_processor.window_size_in_samples = window_size_in_samples
                    # unterscheiden ob mean oder nicht mean
                    if use_mean_sub:
                        emg_processor.normalizer = normalizer_mean
                        emg_processor.mean_rest = mean_rest
                    else:
                        emg_processor.normalizer = normalizer_no_mean
                    emg_processor.grid_aranger = grid_arranger
                    if emg_processor.use_important_channels:
                        if use_mean_sub:
                            emg_processor.channels = important_channels_mean
                            emg_processor.channels_row_shape = channels_row_shape_mean
                        else:
                            emg_processor.channels = important_channels_no_mean
                            emg_processor.channels_row_shape = channels_row_shape_no_mean
                    avg_loss, mse_loss = emg_processor.run(already_build=True)

                if count == 1:
                    normalizer_no_mean = emg_processor.normalizer
                    if emg_processor.use_important_channels:
                        important_channels_no_mean = emg_processor.channels
                        channels_row_shape_no_mean = emg_processor.channels_row_shape

                if count == 0:
                    num_previous_samples = emg_processor.num_previous_samples
                    window_size_in_samples = emg_processor.window_size_in_samples
                    normalizer_mean = emg_processor.normalizer
                    mean_rest = emg_processor.mean_rest
                    grid_arranger = emg_processor.grid_aranger
                    if emg_processor.use_important_channels:
                        important_channels_mean = emg_processor.channels
                        channels_row_shape_mean = emg_processor.channels_row_shape
                count += 1
                print("avg__r2_loss_after_eval: ", avg_loss)
                print("mse_loss_after_eval: ", mse_loss)
                if use_mean_sub:
                    evaluation_results_mean_sub.append(avg_loss)
                    mse_evaluation_results_mean_sub.append(mse_loss)
                else:
                    evaluation_results_no_mean_sub.append(avg_loss)
                    mse_evaluation_results_no_mean_sub.append(mse_loss)

                if best_r2_mean is None:
                    best_r2_mean = avg_loss
                    best_mse_mean = mse_loss
                if best_r2_no_mean is None:
                    best_r2_no_mean = avg_loss
                    best_mse_no_mean = mse_loss

                if use_mean_sub:
                    if avg_loss > best_r2_mean:
                        best_r2_mean = avg_loss
                    if mse_loss < best_mse_mean:
                        best_mse_mean = mse_loss
                else:
                    if avg_loss > best_r2_no_mean:
                        best_r2_no_mean = avg_loss
                    if mse_loss < best_mse_no_mean:
                        best_mse_no_mean = mse_loss

            if use_shallow_conv:
                plt.figure()
                x = [1,10,20,25,35,50,100,125,150,175,200,250,500,1000,1500,2000,2500,3000,4000]
                x = x[:x.index(epochs)+1]
                if len(evaluation_results_mean_sub) == len(x):
                    plt.plot(x,evaluation_results_mean_sub, label="mean_sub",color="red",marker="X")
                if len(evaluation_results_no_mean_sub) == len(x):
                    plt.plot(x,evaluation_results_no_mean_sub,label="no_mean_sub",color="blue",marker="X")
                if len(mse_evaluation_results_mean_sub) == len(x):
                    plt.plot(x, mse_evaluation_results_mean_sub, label="mse_mean_sub", color="red", marker="o")
                if len(mse_evaluation_results_no_mean_sub) == len(x):
                    plt.plot(x, mse_evaluation_results_no_mean_sub, label="mse_no_mean_sub", color="blue", marker="o")

                # Creating a dummy plot element for the additional text
                plt.plot([], [], ' ', label='Best mse no mean is ' + str(round(best_mse_no_mean,3)))
                plt.plot([], [], ' ', label='Best mse mean is ' + str(round(best_mse_mean,3)))
                plt.plot([], [], ' ', label='Best r2 no mean is ' + str(round(best_r2_no_mean,3)))
                plt.plot([], [], ' ', label='Best r2 mean is ' + str(round(best_r2_mean,3)))
                plt.grid()
                plt.ylabel("avg_loss")
                plt.xlabel("epochs")
                plt.title(method)
                plt.legend()

                train_name = emg_processor.patient_id.split("_")[-1]
                test_name = emg_processor.use_recorded_data.split("_")[-1].split("/")[0]
                plt.savefig(r"D:\Lab\MasterArbeit\Plots_Model_Hyperparameters/" + method +  "_" +  train_name + "_" + test_name + "_important_channels.png")


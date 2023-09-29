from sklearn.utils import shuffle
import time
from exo_controller.helpers import *
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#TODO min_samples 5 besser als 30
class MultiDimensionalDecisionTree:
    def __init__(self, important_channels, movements, windom_size=150, num_trees=1, sample_difference_overlap=64, max_depth = 10, min_samples_split=20,
                 num_previous_samples=None):

        self.sample_difference_overlap = sample_difference_overlap
        self.max_depth = max_depth # max depth of the tree
        self.min_samples_split = min_samples_split # min number of samples to split
        self.important_channels = important_channels # all the channels that are important between 0-320
        self.num_channels = len(self.important_channels) # number of channels that are important
        self.num_trees = num_trees # number of trees for one movement and one feature (local,time)
        self.movements = movements # all the movements that we want to classify
        self.num_movements = 0 # number of movements that we want to classify
        self.window_size = windom_size # window size for the data ( length of the window in ms)
        self.window_size_in_samples = int((self.window_size / 1000) * 2048) # window size in samples
        self.sample_difference_overlap = sample_difference_overlap # difference between the start of the next window and the start of the previous window in samples
        self.overlap = self.window_size_in_samples - sample_difference_overlap # overlap between windows in samples
        self.movment_dict = {}
        if num_previous_samples is None:
            num_previous_samples = self.select_default_num_previous()
        self.num_previous_samples = num_previous_samples


        count = 0
        for i in range(len(self.movements)): #check if finger already in dict and add the movement number to the dict
            if self.movements[i] in ["2pinch","3pinch","fist"]:
                continue
            matches = []
            part = self.movements[i].split("_")[0]
            for key in self.movment_dict.keys():
                if part in key:
                    matches.append(key)

            if len(matches) > 0:
                self.movment_dict[self.movements[i]] = self.movment_dict[matches[0]]
            else:
                self.movment_dict[self.movements[i]] = count
                count += 1
                self.num_movements +=1


        self.trees = []
        #for _ in range(self.num_trees * self.num_movements * len(self.num_previous_samples)):
        for _ in range(5):
            #self.trees.append(DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split))
            #self.trees.append(MultiOutputRegressor(DecisionTreeRegressor( min_samples_split=self.min_samples_split)))
            # decrease C to reduce overfitting
            # increase C to better fit the dat
            # epsilo = margin of tolerance where no penalty is assigned to errors
            # low epsilon = more overfitting
            #self.trees.append(MultiOutputRegressor(SVR(kernel="linear",C=0.5,epsilon=0.2)))
            self.trees.append(MultiOutputRegressor(RandomForestRegressor(n_estimators=30, min_samples_split=self.min_samples_split, min_samples_leaf=15)))

    def select_default_num_previous(self,middle_time_difference = 100):

        overlap_in_time = self.sample_difference_overlap / 2048
        middle_time_difference_in_samples = middle_time_difference/1000
        middle_selected_default = middle_time_difference_in_samples/ overlap_in_time
        self.num_previous_samples = [int(middle_selected_default/2),int(middle_selected_default),int(middle_selected_default*2),int(middle_selected_default*3)]
        print("Selected default number of previous samples: ",self.num_previous_samples)
        return self.num_previous_samples

    def predict(self, X,tree_numbers):
        """
        Predict the target values using the list of trained models and average the results.

        :param X: A 2D numpy array of samples for prediction.

        :return: Averaged predictions.
        """
        if not self.trees:
            raise Exception("No models trained yet!")

        # Collect predictions from all models
        predictions = []
        for i in tree_numbers:
            predictions.append(self.trees[i].predict(X))
        predictions = np.array(predictions)
        for i in range(len(predictions)):
            print("Tree number: ",i, "Prediction: ",predictions[i])
        # Average the predictions
        averaged_predictions = predictions.mean(axis=0)
        print("Averaged prediction: ",averaged_predictions)

        return averaged_predictions

    def build_training_data(self, movement_names, path_to_subject_dat, window_size=int((150 / 1000) * 2048),
                            overlap=int((150 / 1000) * 2048) - 150, split_ratio=0.8):
        """
        This function builds the training data for the random forest
        the format for one input value should be [[movement1, movement2, ....][value1, value2, ....]] for ref labels
        and [320x1] for emg data
        :param movement_names:
        :param path_to_subject_dat:
        :param window_size:
        :param overlap:
        :param split_ratio:
        :return:
        """
        segments = []
        labels = []

        for movement in tqdm.tqdm(movement_names,desc="Building training data for local differences"):
            emg_data, Mu_data, ref_data = open_all_files_for_one_patient_and_movement_(path_to_subject_dat, movement)
            ref_erweitert = np.zeros((self.num_movements,len(ref_data))) # [num_movements x num_samples]
            if movement != "2pinch":
                ref_erweitert[ref_erweitert == 0.0] = 0.0
            ref_data = normalize_2D_array(ref_data,axis=0)
            if movement != "2pinch":
                ref_erweitert[self.movment_dict[movement],:] = ref_data[:,0] # jetzt werte für die bewegung an passenden index eintragen für anderen finger einträge auf 0.5 setzen
            else:
                for k in range(2):
                    ref_erweitert[k, :] = ref_data[:, k]
            emg_data = reshape_emg_channels_into_320(emg_data)
            emg_data= emg_data[self.important_channels,:]
            for i in range(0, len(emg_data[0]) - window_size + 1, window_size - overlap): # da unterschiedliche länge von emg und ref nur machen wenn ref noch nicht zuzende ist
                if i <= ref_data.shape[0]:
                    segment = calculate_emg_rms_row(emg_data,i,self.window_size_in_samples)
                    segment = normalize_2D_array(segment)
                    # feature = calculate_rms(segment)
                    # segments.append(feature)
                    label = ref_erweitert[:,i]
                    segments.append(segment)
                    labels.append(label)


        segments, labels = shuffle(segments, labels, random_state=42)
        split_index = int(len(labels) * split_ratio)
        self.X_train_local, self.X_test_local = segments[:split_index], segments[split_index:]
        self.y_train_local, self.y_test_local = labels[:split_index], labels[split_index:]



        ############# For the time traind data #############
        results = []

        for idx in tqdm.tqdm(self.num_previous_samples,desc="Building training data for time differences"):
            combined_diffs = []
            combined_ys = []

            for movement in movement_names:
                emg_data, Mu_data, ref_data = open_all_files_for_one_patient_and_movement(path_to_subject_dat, movement)
                ref_erweitert = np.zeros((self.num_movements, len(ref_data)))  # [num_movements x num_samples]
                if movement != "2pinch":
                    ref_erweitert[ref_erweitert == 0.0] = 0.0
                ref_data = normalize_2D_array(ref_data,axis=0)
                if movement != "2pinch":
                    ref_erweitert[self.movment_dict[movement], :] = ref_data[:,0]  # jetzt werte für die bewegung an passenden index eintragen für anderen finger einträge auf 0.5 setzen
                else:
                    for k in range(2):
                        ref_erweitert[k, :] = ref_data[:, k] # TODO maybe change back to k if do not want both values to be the same
                emg_data = reshape_emg_channels_into_320(emg_data)
                emg_data = emg_data[self.important_channels, :]
                for i in range(0, len(emg_data[0]) - window_size + 1, window_size - overlap):  # da unterschiedliche länge von emg und ref nur machen wenn ref noch nicht zuzende ist
                    if (i <= ref_data.shape[0]) and ( i-idx >= 0):
                        heatmap = calculate_emg_rms_row(emg_data, i, self.window_size_in_samples)
                        heatmap = normalize_2D_array(heatmap)
                        previous_heatmap = calculate_emg_rms_row(emg_data, i-idx, self.window_size_in_samples)
                        previous_heatmap = normalize_2D_array(previous_heatmap)
                        difference_heatmap = np.abs(np.subtract(heatmap , previous_heatmap))
                        #difference_heatmap = normalize_2D_array(difference_heatmap)
                        #difference_heatmap = normalize_2D_array(calculate_emg_rms_row(emg_data, i,idx *(self.window_size_in_samples-self.sample_difference_overlap)))
                        label = ref_erweitert[:, i]
                        combined_diffs.append(difference_heatmap)
                        combined_ys.append(label)

            # Convert combined differences and y values to numpy arrays
            combined_diffs = np.array(combined_diffs)
            combined_ys = np.array(combined_ys)

            # Shuffle and split the combined data for the current index
            combined_diffs, combined_ys = shuffle(combined_diffs, combined_ys, random_state=42)
            split_index = int(len(combined_diffs) * split_ratio)
            X_train, X_test = combined_diffs[:split_index], combined_diffs[split_index:]
            y_train, y_test = combined_ys[:split_index], combined_ys[split_index:]

            results.append((X_train, X_test, y_train, y_test))
            self.training_data_time = results

    def load_trainings_data(self):
        self.X_test_local = np.array(load_pickle_file( r"D:\Lab\MasterArbeit\trainings_data\X_test_local_" +str(self.window_size) + "_" + str(self.sample_difference_overlap)+ ".pkl"))
        self.y_test_local = np.array(load_pickle_file( r"D:\Lab\MasterArbeit\trainings_data\y_test_local_" +str(self.window_size) + "_" + str(self.sample_difference_overlap)+ ".pkl" ))
        self.X_train_local = np.array(load_pickle_file(r"D:\Lab\MasterArbeit\trainings_data\X_train_local_" +str(self.window_size) + "_" + str(self.sample_difference_overlap)+ ".pkl"))
        self.y_train_local = np.array(load_pickle_file(r"D:\Lab\MasterArbeit\trainings_data\y_train_local_" +str(self.window_size) + "_" + str(self.sample_difference_overlap)+ ".pkl" ))
        self.training_data_time = load_pickle_file( r"D:\Lab\MasterArbeit\trainings_data\training_data_time_" +str(self.window_size) + "_" + str(self.sample_difference_overlap)+ ".pkl")


    def add_data_for_local_detection(self,x_train,y_train,x_test,y_test):
        self.X_train_local = x_train
        self.y_train_local = y_train
        self.X_test_local = x_test
        self.y_test_local = y_test

    def add_data_for_time_detection(self,x_train,y_train,x_test,y_test):
        self.X_train_time = x_train
        self.y_train_time = y_train
        self.X_test_time = x_test
        self.y_test_time = y_test


    def normalize_emg(self, emg_data):
        return normalize_2D_array(emg_data)

    def train(self):
        start = time.time()
        for i in tqdm.tqdm(range(len(self.trees)),desc="Training trees"):
            if i == 0:
                self.trees[i].fit(np.array(self.X_train_local),np.array(self.y_train_local))
            else:
                self.trees[i].fit(self.training_data_time[i-1][0],self.training_data_time[i-1][2])
        print("Training Time: %s seconds" % (str(time.time() - start)))

    def evaluate(self):
        for i in tqdm.tqdm(range(len(self.trees)), desc="Evaluating trees"):
            if i == 0:
                res = self.trees[i].predict(self.X_test_local)
                truth = self.y_test_local
                mae = mean_absolute_error(truth, res, multioutput='raw_values')
                mse = mean_squared_error(truth, res, multioutput='raw_values')
                r2 = r2_score(truth, res, multioutput='raw_values')

                print("local Mean Absolute Error: ", mae)
                print("local Mean Squared Error : ", mse)
                print("local R^2 Score : ", r2)


            else:
                res = self.trees[i].predict(self.training_data_time[i - 1][1])
                truth = self.training_data_time[i - 1][3]
                mae = mean_absolute_error(truth, res, multioutput='raw_values')
                mse = mean_squared_error(truth, res, multioutput='raw_values')
                r2 = r2_score(truth, res, multioutput='raw_values')

                print("time Mean Absolute Error for tree " +str(i)+" : ", mae)
                print("time Mean Squared Error for tree " +str(i)+" : ", mse)
                print("time R^2 Score for tree " +str(i)+" : ", r2)

    def save_trainings_data(self):

        save_as_pickle(self.X_test_local, r"D:\Lab\MasterArbeit\trainings_data\X_test_local_" + str(self.window_size) + "_" + str(self.sample_difference_overlap) + ".pkl")
        save_as_pickle(self.y_test_local,r"D:\Lab\MasterArbeit\trainings_data\y_test_local_" + str(self.window_size) + "_" + str(self.sample_difference_overlap) + ".pkl" )
        save_as_pickle(self.X_train_local, r"D:\Lab\MasterArbeit\trainings_data\X_train_local_" + str(self.window_size) + "_" + str(self.sample_difference_overlap) + ".pkl")
        save_as_pickle(self.y_train_local, r"D:\Lab\MasterArbeit\trainings_data\y_train_local_" + str(self.window_size) + "_" + str(self.sample_difference_overlap) + ".pkl")
        save_as_pickle(self.training_data_time,r"D:\Lab\MasterArbeit\trainings_data\training_data_time_" + str(self.window_size) + "_" + str(self.sample_difference_overlap) + ".pkl")
    def simulate_realtime_prediction(self):
        for i in tqdm.tqdm(range(len(self.trees)),desc="Evaluating trees"):
            if i == 0:
                res = self.trees[i].predict(self.X_test_local)
                truth = self.y_test_local
                score = np.mean(np.abs(np.subtract(res,self.y_test_local)),axis=0)
                print("local differences for tree " + str(i)+ " : ", (1 - score[0]) * 100,(1 - score[1]) * 100)
            else:
                res = self.trees[i].predict(self.training_data_time[i-1][1])
                truth = self.training_data_time[i-1][3]
                score = np.mean(np.abs(np.subtract(res, self.training_data_time[i-1][3])),axis=0)
                print("time differences for tree " + str(i)+ " : ", (1 - score[0]) * 100,(1 - score[1]) * 100)

    def visualize_trees(self):
        for i in range(len(self.trees)):
            print("Tree number: ",i)
            print("Tree depth: ",self.trees[i].get_depth())
            print("Number of leaves: ",self.trees[i].get_n_leaves())
            print("Number of samples: ",self.trees[i].get_n_leaves())
            print("Feature importance: ",self.trees[i].feature_importances_)
            print("Number of features: ",self.trees[i].n_features_)
            print("Number of outputs: ",self.trees[i].n_outputs_)
            print("Number of classes: ",self.trees[i].n_classes_)
            print("Number of samples: ",self.trees[i].n_samples_)
            print("Number of features: ",self.trees[i].n_features_)
            print("Number of features: ",self.trees[i].n_features_)

if __name__ == "__main__":
    #movements = ["thumb_slow", "thumb_fast", "index_slow", "index_fast","2pinch"]
    movements = ["thumb_slow",  "index_slow", "2pinch"]
    # important_channels = extract_important_channels(movements, r"D:\Lab\data\extracted\Sub2")
    # print("there were following number of important channels found: ",len(important_channels))
    # channels = []
    # for important_channel in important_channels:
    #     channels.append(from_grid_position_to_row_position(important_channel))
    # print("there were following number of important channels found: ",len(channels))

    channels= range(320)
    model = MultiDimensionalDecisionTree(important_channels=channels,movements=movements)
    model.build_training_data(model.movements, r"D:\Lab\data\extracted\Sub2")
    #odel.load_trainings_data()
    model.save_trainings_data()
    model.train()
    model.evaluate()



    # Thumb = thumb0 y
    # index = index0 x



    #TODO ich schaue mir bei jeem knoten die noch verbleibenden samples an (data) , die in diesen bereich fallen

        # mache zweiten baum der die unterscheidung nicht macht durch berechnen mean von emg und bweichung von mean
        # sondern nehme emg signale und berechne für alle aufeinanderfolgenden samples die änderung zwischen diesem samole und sample davor (oder 3 oder what ever davor)
        # heatmap differenz aber ich muss vorzeichen behalten ( nicht abs )
        # dann threshold berechnen wie bei emg nur halt mit zeitlicher differenz zwischen aufeinanderfolgenden samples
        # dann dataset verringern durch
        # links = alle emg data bei denen die differenz zwischen zwei aufeinanderfolgenden samples kleiner ist als threshold wenn t und t+1 kleiner als threshold dann beide hinzu
        # dann so weiter bis auch wieder nicht mind. 20 samples in bereich fallen

        # für mehrere vreschiedene zeiten zwischen emg heatmap bäume machen und für mehrere emg bäume
        # dann meinen algo von bachelorarbeit verwenden um zu smoothen...


        #nicht nur eindimensional die heatmaps anschauen... ich muss auch schauen wie sich die heatmaps über die zeit verändern und wie sie bei den nachbarn ist
        # was ist wenn ich pca auf die Heatmaps bzw Trainingsdaten mache =====?


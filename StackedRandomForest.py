import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time
from exo_controller.helpers import *
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import Ridge


#TODO min_samples 5 besser als 30
class MultiDimensionalDecisionTree:
    def __init__(self, important_channels, movements, mean_heatmaps= None,windom_size=150, num_trees=40, sample_difference_overlap=64, max_depth = 15, min_samples_split=5,
                 num_previous_samples=None,classification = False):
        self.classification = classification # whether to use classification or regression
        self.mean_heatmaps = mean_heatmaps
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
            #self.trees.append(MultiOutputRegressor(
            #    Ridge()))
            if self.classification:
                self.trees.append(RandomForestClassifier(n_estimators=self.num_trees, min_samples_split=self.min_samples_split, max_depth=self.max_depth,warm_start=True))
            else:
                self.trees.append(MultiOutputRegressor(RandomForestRegressor(n_estimators=self.num_trees, min_samples_split=self.min_samples_split,max_depth=self.max_depth,warm_start=True)))

    def compare_predictions(self, predictions, truth,tree_number=None):
        plot_predictions(predictions, truth)

    def select_default_num_previous(self):
        """
        Select the default number of previous samples to use for the time difference model.
        We take the windows size (f.e 150 ms) and calculate the number of samples for 1, 2, 3 and 4 times the window size.
        later we will be at a point in time t and make the difference heatmap between t and t-1*(window_size_in_samples), t-2*(window_size_in_samples), t-3*(window_size_in_samples) and t-4*(window_size_in_samples)
                """
        self.num_previous_samples = [int(self.window_size_in_samples / 2), int(self.window_size_in_samples),
                                     int(self.window_size_in_samples * 2), int(self.window_size_in_samples * 3)]
        print("Selected default number of previous samples: ", self.num_previous_samples)
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

    def build_training_data(self, movement_names, path_to_subject_dat, split_ratio=0.85):
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
        res= {}
        segments = []
        labels = []

        window_size = self.window_size_in_samples

        for movement in tqdm.tqdm(movement_names,desc="Building training data for local differences"):
            heatmaps = []
            emg_data, Mu_data, ref_data = open_all_files_for_one_patient_and_movement(path_to_subject_dat, movement)
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
            for i in range(0, len(emg_data[0]) - window_size + 1, self.sample_difference_overlap): # da unterschiedliche länge von emg und ref nur machen wenn ref noch nicht zuzende ist
                if i <= ref_data.shape[0]:
                    segment = calculate_emg_rms_row(emg_data,i,self.window_size_in_samples)
                    segment = normalize_2D_array(segment)
                    # feature = calculate_rms(segment)
                    # segments.append(feature)
                    if self.classification:
                        maxima,minima = get_locations_of_all_maxima(ref_data[:,0])
                        belongs_to_movement,_ = check_to_which_movement_cycle_sample_belongs(i,maxima,minima)
                        label = belongs_to_movement
                        # plt.figure()
                        # plt.plot(ref_data[:,0])
                        # plt.title("belong to movement: " + str(belongs_to_movement))
                        # plt.scatter(i,ref_data[i,0],c="red")
                        # plt.scatter(maxima,ref_data[maxima,0],c="green")
                        # plt.scatter(minima,ref_data[minima,0],c="blue")
                        # #plt.xlim(i,i+20000)
                        # plt.show()
                    else:
                        label = ref_erweitert[:,i]
                    segments.append(segment)
                    labels.append(label)


            # segments, labels = shuffle(segments, labels, random_state=42)
            # split_index = int(len(labels) * split_ratio)
            # X_train_local, X_test_local = segments[:split_index], segments[split_index:]
            # y_train_local, y_test_local = labels[:split_index], labels[split_index:]
            # dict = {"X_train_local":X_train_local,"X_test_local":X_test_local,"y_train_local":y_train_local,"y_test_local":y_test_local}
            # res[movement] = dict
            # self.res = res

        split_index = int(len(labels) * split_ratio)
        segments, labels = shuffle(segments, labels, random_state=42)
        print("number of samples generated for local: : ", len(labels))
        self.X_train_local, self.X_test_local = segments[:split_index], segments[split_index:]
        self.y_train_local, self.y_test_local = labels[:split_index], labels[split_index:]





        ############ For the time traind data #############
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
                for i in range(0, len(emg_data[0]) - window_size + 1, self.sample_difference_overlap):  # da unterschiedliche länge von emg und ref nur machen wenn ref noch nicht zuzende ist
                    if (i <= ref_data.shape[0]) and ( i-idx >= 0):
                        heatmap = calculate_emg_rms_row(emg_data, i, self.window_size_in_samples)
                        heatmap = normalize_2D_array(heatmap)
                        previous_heatmap = calculate_emg_rms_row(emg_data, i-idx, self.window_size_in_samples)
                        previous_heatmap = normalize_2D_array(previous_heatmap)
                        #difference_heatmap = np.abs(np.subtract(heatmap , previous_heatmap))
                        difference_heatmap = normalize_2D_array(np.subtract(heatmap, previous_heatmap))
                        #difference_heatmap = normalize_2D_array(difference_heatmap)
                        #difference_heatmap = normalize_2D_array(calculate_emg_rms_row(emg_data, i,idx *(self.window_size_in_samples-self.sample_difference_overlap)))


                        if self.classification:
                            maxima, minima = get_locations_of_all_maxima(ref_data[:, 0])
                            belongs_to_movement, _ = check_to_which_movement_cycle_sample_belongs(i, maxima, minima)
                            label = belongs_to_movement
                        else:
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
        # start = time.time()
        # for i in tqdm.tqdm(range(len(self.trees)),desc="Training trees"):
        #     #self.trees[i].fit(np.array(self.X_train_local),np.array(self.y_train_local))
        #     self.trees[i].fit(np.array(self.res[self.movements[i]]["X_train_local"]), np.array(self.res[self.movements[i]]["y_train_local"]))
        # print("Training Time: %s seconds" % (str(time.time() - start)))

        start = time.time()
        for i in tqdm.tqdm(range(len(self.trees)), desc="Training trees"):
            if i == 0:
                self.trees[i].fit(np.array(self.X_train_local), np.array(self.y_train_local))
            else:
                self.trees[i].fit(self.training_data_time[i - 1][0], self.training_data_time[i - 1][2])
        print("Training Time: %s seconds" % (str(time.time() - start)))

    def get_heatmap_of_this_sample(self,sample_number):
        pass
    def which_mean_heatmap_is_corr_biggest(self,current_heatmap):
        pass
    def evaluate(self):
        # number_of_times_right_tree_chosen = 0
        # count = 0
        # for movement in tqdm.tqdm(range(len(self.movements)),desc="Evaluating trees"):
        #     truth = self.res[self.movements[movement]]["y_test_local"]
        #     results = []
        #     res = self.trees[movement].predict(self.res[self.movements[movement]]["X_test_local"])
        #     mae = mean_absolute_error(truth, res, multioutput='raw_values')
        #     mse = mean_squared_error(truth, res, multioutput='raw_values')
        #     r2 = r2_score(truth, res, multioutput='raw_values')
        #     print("local Mean Absolute Error perfect fit: ", mae)
        #     print("local Mean Squared Error perfect fit: ", mse)
        #     print("local R^2 Score perfect fit: ", r2)
        #
        #     for j in range(len(results)):
        #         for k in range(len(results[j])):
        #             count +=1
        #
        #
        #         mae = mean_absolute_error(truth, res, multioutput='raw_values')
        #         mse = mean_squared_error(truth, res, multioutput='raw_values')
        #         r2 = r2_score(truth, res, multioutput='raw_values')
        #
        #
        #
        #         print("local Mean Absolute Error: ", mae)
        #         print("local Mean Squared Error : ", mse)
        #         print("local R^2 Score : ", r2)
        for i in tqdm.tqdm(range(len(self.trees)), desc="Evaluating trees"):
            if i == 0:
                res = self.trees[i].predict(self.X_test_local)
                truth = self.y_test_local
                self.compare_predictions(res, truth,tree_number= i)
                if not self.classification:
                    mae = mean_absolute_error(truth, res, multioutput='raw_values')
                    mse = mean_squared_error(truth, res, multioutput='raw_values')
                    r2 = r2_score(truth, res, multioutput='raw_values')

                    print("local Mean Absolute Error: ", mae)
                    print("local Mean Squared Error : ", mse)
                    print("local R^2 Score : ", r2)
                else:
                    print("local accuracy: ", accuracy_score(truth, res))
                    print("local precision: ", precision_score(truth, res, average='weighted'))
                    print("local recall: ", recall_score(truth, res, average='weighted'))

            else:
                res = self.trees[i].predict(self.training_data_time[i - 1][1])
                truth = self.training_data_time[i - 1][3]
                self.compare_predictions(res, truth,tree_number= i)

                if not self.classification:
                    mae = mean_absolute_error(truth, res, multioutput='raw_values')
                    mse = mean_squared_error(truth, res, multioutput='raw_values')
                    r2 = r2_score(truth, res, multioutput='raw_values')

                    print("time Mean Absolute Error for tree " + str(i) + " : ", mae)
                    print("time Mean Squared Error for tree " + str(i) + " : ", mse)
                    print("time R^2 Score for tree " + str(i) + " : ", r2)
                else:
                    print("time accuracy for tree " + str(i) + " : ", accuracy_score(truth, res))
                    print("time precision for tree " + str(i) + " : ", precision_score(truth, res, average='weighted'))
                    print("time recall for tree " + str(i) + " : ", recall_score(truth, res, average='weighted'))

        for i in tqdm.tqdm(range(len(self.trees)-1), desc="Evaluating trees"):
            res_local = self.trees[0].predict(self.X_test_local)
            truth = self.y_test_local
            pred_time_tree = self.trees[i+1].predict(self.training_data_time[i][1])
            combined=  (res_local + pred_time_tree)/2

            if not self.classification:
                res= combined
                self.compare_predictions(res, truth, tree_number=i)
                mae = mean_absolute_error(truth, res, multioutput='raw_values')
                mse = mean_squared_error(truth, res, multioutput='raw_values')
                r2 = r2_score(truth, res, multioutput='raw_values')

                print("combined Mean Absolute Error for tree " + str(i) + " : ", mae)
                print("combined Mean Squared Error for tree " + str(i) + " : ", mse)
                print("combined R^2 Score for tree " + str(i) + " : ", r2)
            else:
                res = int(combined)
                self.compare_predictions(res, truth, tree_number=i)
                print("combined accuracy for tree " + str(i) + " : ", accuracy_score(truth, res))
                print("combined precision for tree " + str(i) + " : ", precision_score(truth, res, average='weighted'))
                print("combined recall for tree " + str(i) + " : ", recall_score(truth, res, average='weighted'))



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

    def visualize_tree(self):
        from sklearn.tree import export_graphviz
        import pydot
        estimator = self.trees[0].estimators_[5]
        export_graphviz(estimator, out_file='tree.dot',
                        feature_names=['x', 'y'],
                        class_names=['0', '1', '2'],
                        rounded=True, proportion=False,
                        precision=2, filled=True)

        (graph,) = pydot.graph_from_dot_file('tree.dot')
        graph.write_png('tree.png')

if __name__ == "__main__":
    #movements = ["thumb_slow", "thumb_fast", "index_slow", "index_fast","2pinch"]
    movements = ["thumb_slow",  "index_slow", "2pinch"]

    #mean_heatmaps = extract_mean_heatmaps(movements, r"D:\Lab\data\extracted\Sub2")

    # important_channels = extract_important_channels(movements, r"D:\Lab\data\extracted\Sub2")
    # print("there were following number of important channels found: ",len(important_channels))
    # channels = []
    # for important_channel in important_channels:
    #    channels.append(from_grid_position_to_row_position(important_channel))
    # print("there were following number of important channels found: ",len(channels))

    channels= range(320)
    model = MultiDimensionalDecisionTree(important_channels=channels,movements=movements,mean_heatmaps=None,classification= False)
    #model.build_training_data(model.movements, r"D:\Lab\data\extracted\Sub2")
    model.load_trainings_data()
    #model.save_trainings_data()
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


from sklearn.utils import shuffle
import time
from exo_controller.helpers import *
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


#TODO min_samples 5 besser als 30
class MultiDimensionalDecisionTree:
    def __init__(self, important_channels, movements,emg,ref,patient_number, windom_size=200, num_trees=50, sample_difference_overlap=64, max_depth = 320, min_samples_split=20,
                 num_previous_samples=None):
        self.emg_data= emg
        self.patient_number = patient_number
        self.ref_data = ref
        self.sample_difference_overlap = sample_difference_overlap # amount of new datapoints coming in
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
        self.overlap = self.window_size_in_samples - sample_difference_overlap # overlap between windows in samples ( amount of still the same)
        self.movment_dict = {}
        if num_previous_samples is None:
            num_previous_samples = self.select_default_num_previous()
        self.num_previous_samples = num_previous_samples
        self.neuromuscular_delay = 50  # neuromuscucular delay in miliseconds
        self.neuromuscular_delay_in_samples = int(
            (self.neuromuscular_delay / 1000) * 2048)  # neuromuscucular delay in samples

        count = 0
        for i in range(len(self.movements)): #check if finger already in dict and add the movement number to the dict
            if self.movements[i] in ["2pinch","3pinch","fist","rest"]:
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
            self.trees.append(MultiOutputRegressor(RandomForestRegressor(n_estimators=self.num_trees, min_samples_split=self.min_samples_split,max_depth=self.max_depth,warm_start=True,n_jobs=-1)))

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

    def save_model(self,  subject):
        if not os.path.exists(r"D:\Lab\MasterArbeit\trainings_data\trained_models/" + subject):
            os.makedirs(r"D:\Lab\MasterArbeit\trainings_data\trained_models/" + subject)
        for i, model in enumerate(self.trees):
            filename = f'model_{i}.pkl'
            joblib.dump(model, r"D:\Lab\MasterArbeit\trainings_data\trained_models/"+ subject + "/" +filename)


    def load_model(self, subject):

        trees = []
        for i in os.listdir(r"D:\Lab\MasterArbeit\trainings_data\trained_models/"+ subject ):
            trees.append(joblib.load(r"D:\Lab\MasterArbeit\trainings_data\trained_models/"+ subject + "/" +i))
        self.trees = trees

    def predict(self, X ,tree_numbers):
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
        # for i in range(len(predictions)):
        #     print("Tree number: ",i, "Prediction: ",predictions[i])
        # Average the predictions
        averaged_predictions = predictions.mean(axis=0)
        print("Averaged prediction: ",averaged_predictions)

        return averaged_predictions

    def build_training_data(self, movement_names, split_ratio=0.9):
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

        window_size = self.window_size_in_samples


        for movement in tqdm.tqdm(movement_names,desc="Building training data for local differences"):

            ref_erweitert = np.zeros((self.num_movements,len(self.ref_data[movement]))) # [num_movements x num_samples]
            if movement != "2pinch":
                ref_erweitert[ref_erweitert == 0.0] = 0.0

            if movement != "rest":
                ref_data = normalize_2D_array(self.ref_data[movement],axis=0)
            else:
                ref_data = self.ref_data[movement]

            if (movement != "2pinch") and (movement != "rest"):
                ref_erweitert[self.movment_dict[movement],:] = ref_data[:,0] # jetzt werte für die bewegung an passenden index eintragen für anderen finger einträge auf 0.5 setzen
            else: # in 2 pinch case
                # thumb has to be 0.45 and index 0.6
                for k in range(2):
                    ref_erweitert[k, :] = ref_data[:, k]
                ref_erweitert[0,:] = np.multiply(ref_erweitert[0,:], 0.45)
                ref_erweitert[1,:]= np.multiply(ref_erweitert[1, :], 0.6)
            emg_data= self.emg_data[movement][self.important_channels,:]
            for i in range(0, len(emg_data[0]) - window_size + 1, self.sample_difference_overlap): # da unterschiedliche länge von emg und ref nur machen wenn ref noch nicht zuzende ist
                if i <= ref_data.shape[0]:
                    segment = calculate_emg_rms_row(emg_data,i,self.window_size_in_samples)
                    segment = normalize_2D_array(segment)
                    # feature = calculate_rms(segment)
                    # segments.append(feature)
                    label = ref_erweitert[:,i]

                    # after the following will be the additional comparison between the current heatmap and the reference signal some time ago or in the future
                    # best would be to take the ref from the signal because first comes the emg signal(heatmap) and the comes the reference or the real output
                    if ((i + self.neuromuscular_delay_in_samples) < ref_erweitert[0].shape[0]):
                        for skip in range(64, self.neuromuscular_delay_in_samples, 64):
                            ref_in_the_future = ref_erweitert[:, i + skip]
                            segments.append(segment)
                            labels.append(ref_in_the_future)

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

                ref_erweitert = np.zeros((self.num_movements, len(self.ref_data[movement])))  # [num_movements x num_samples]
                if movement != "2pinch":
                    ref_erweitert[ref_erweitert == 0.0] = 0.0
                if movement != "rest":
                    ref_data = normalize_2D_array(self.ref_data[movement], axis=0)
                else:
                    ref_data = self.ref_data[movement]

                if (movement != "2pinch") and (movement != "rest"):
                    ref_erweitert[self.movment_dict[movement], :] = ref_data[:,0]  # jetzt werte für die bewegung an passenden index eintragen für anderen finger einträge auf 0.5 setzen
                else:
                    for k in range(2):
                        ref_erweitert[k, :] = ref_data[:, k] # TODO maybe change back to k if do not want both values to be the same
                emg_data = self.emg_data[movement][self.important_channels, :]
                for i in range(0, len(emg_data[0]) - window_size + 1, self.sample_difference_overlap):  # da unterschiedliche länge von emg und ref nur machen wenn ref noch nicht zuzende ist
                    if (i <= ref_data.shape[0]) and ( i-idx >= 0):
                        heatmap = calculate_emg_rms_row(emg_data, i, self.window_size_in_samples)
                        heatmap = normalize_2D_array(heatmap)
                        previous_heatmap = calculate_emg_rms_row(emg_data, i-idx, self.window_size_in_samples)
                        previous_heatmap = normalize_2D_array(previous_heatmap)
                        difference_heatmap = normalize_2D_array(np.subtract(heatmap, previous_heatmap))
                        label = ref_erweitert[:, i]

                        # after the following will be the additional comparison between the current heatmap and the reference signal some time ago or in the future
                        # best would be to take the ref from the signal because first comes the emg signal(heatmap) and the comes the reference or the real output
                        if ((i + self.neuromuscular_delay_in_samples) < ref_erweitert[0].shape[0]):
                            for skip in range(64, self.neuromuscular_delay_in_samples, 64):
                                ref_in_the_future = ref_erweitert[:, i + skip]
                                combined_diffs.append(difference_heatmap)
                                combined_ys.append(ref_in_the_future)

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

    def compare_predictions(self, predictions, truth, tree_number=None, realtime=False):
        plot_predictions(truth, predictions, tree_number=tree_number, realtime=realtime)
    def load_trainings_data(self):
        self.X_test_local = np.array(load_pickle_file( r"trainings_data/resulting_trainings_data/subject_"+str(self.patient_number)+"/X_test_local.pkl"))
        self.y_test_local = np.array(load_pickle_file( r"trainings_data/resulting_trainings_data/subject_"+str(self.patient_number)+"/y_test_local.pkl"))
        self.X_train_local = np.array(load_pickle_file(r"trainings_data/resulting_trainings_data/subject_"+str(self.patient_number)+"/X_train_local.pkl"))
        self.y_train_local = np.array(load_pickle_file(r"trainings_data/resulting_trainings_data/subject_"+str(self.patient_number)+"/y_train_local.pkl"))
        self.training_data_time = load_pickle_file( r"trainings_data/resulting_trainings_data/subject_"+str(self.patient_number)+"/training_data_time.pkl")


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


    def train(self):
        start = time.time()
        for i in tqdm.tqdm(range(len(self.trees)),desc="Training trees"):
            if i == 0:
                self.trees[i].fit(np.array(self.X_train_local),np.array(self.y_train_local))
            else:
                self.trees[i].fit(self.training_data_time[i-1][0],self.training_data_time[i-1][2])
        print("Total Training Time: %s seconds" % (str(time.time() - start)))

    def evaluate(self,give_best_time_tree=False):
        best_time_tree = -1
        r2_of_best_time_tree = -1
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
                if give_best_time_tree:
                    if r2[0]+ r2[1] > r2_of_best_time_tree:
                        best_time_tree = i
                        r2_of_best_time_tree = r2[0] + r2[1]

                print("time Mean Absolute Error for tree " +str(i)+" : ", mae)
                print("time Mean Squared Error for tree " +str(i)+" : ", mse)
                print("time R^2 Score for tree " +str(i)+" : ", r2)

        if give_best_time_tree:
            print("best time tree: ", best_time_tree)
            print("r2 of best time tree: ", r2_of_best_time_tree/2)
            return best_time_tree
    def save_trainings_data(self):


        save_as_pickle(self.X_test_local, r"trainings_data/resulting_trainings_data/subject_" + str(self.patient_number) + "/X_test_local.pkl")
        save_as_pickle(self.y_test_local,r"trainings_data/resulting_trainings_data/subject_" + str(self.patient_number) + "/y_test_local.pkl")
        save_as_pickle(self.X_train_local,r"trainings_data/resulting_trainings_data/subject_" + str(self.patient_number) + "/X_train_local.pkl")
        save_as_pickle(self.y_train_local,  r"trainings_data/resulting_trainings_data/subject_" + str(self.patient_number) + "/y_train_local.pkl")
        save_as_pickle(self.training_data_time,r"trainings_data/resulting_trainings_data/subject_" + str(self.patient_number) + "/training_data_time.pkl")
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
    model = MultiDimensionalDecisionTree(important_channels=channels,movements=movements,emg=None,ref=None,patient_number=2)
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


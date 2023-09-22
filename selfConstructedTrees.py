from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tqdm
import time
from helpers import *
import xgboost as xgb

# class MultiDimensionalDecisionTree:
#     def __init__(self, buffer_size=5, min_samples_split=20, max_depth=5):
#         self.min_samples_split = min_samples_split
#         self.max_depth = max_depth
#         self.buffer_size = buffer_size
#         self.emg_buffer = []
#         self.label_buffer = []
#         self.tree = None
#         self.threshold = 0.5
#         self.root =

class TreeNode:
    def __init__(self, is_root=False,num_previous_samples = 0):
        self.thresholds = None
        self.children = []
        self.parent = None
        self.is_leaf = False
        self.is_root = is_root
        self.previous_samples = None
        self.num_previous_samples = num_previous_samples
        self.thresholds_previous_samples = None

    def set_parent(self, node):
        self.parent = node

    def get_parent(self):
        return self.parent

    def add_child(self, thresholds):
        child = TreeNode(thresholds, num_previous_samples=self.num_previous_samples)
        child.set_parent(self)
        self.children.append(child)
        return child  # Return child node for further operations if needed

    def find_best_splits(self, X):
        # X = # num chunks x #num_channels x 1
        # y = #num_chunks x #nu_movements x 1
        best_splits = []
        # TODO hier unterschieden um num previous_samples >0  ja dann muss anders machen : muss zeitabstände berecchnen und davpn threshold setzen
        # TODO not only take threshold as value of one point but as mean of two neighbouring points ( not time neighbours but value neighbours)
        for channel in range(X.shape[1]):
            best_split_value = 0
            best_differences_from_samples_to_splitvalue = 0
            for i in range(X.shape[0]):
                if i > 0:
                    split_value = X[i, channel]
                    differences_from_samples_to_splitvalue = np.sum(np.square(np.subtract(X[:, channel] , split_value))) # distance of all other samples to this threshold
                    if best_differences_from_samples_to_splitvalue > differences_from_samples_to_splitvalue:
                        best_split_value = split_value
                        best_differences_from_samples_to_splitvalue = differences_from_samples_to_splitvalue
                else: #for initialisation
                    best_split_value = X[i, channel]
                    best_differences_from_samples_to_splitvalue = np.sum(np.square(np.subtract(X[:, channel] , X[i, channel])))
            best_splits.append(best_split_value)
        self.thresholds = best_splits

        # if we also want to consider how the heatmaps changed over time we take one sample after another and calculate the differences between them

        if self.num_previous_samples > 0:
            difference_heatmap = np.abs(np.subtract(self.previous_x[1:],self.previous_x[:-1])
            best_splits_previous_samples = [] # has size: #previous_sample_to_watch, value between the two following samples
            for i in range (self.previous_x.shape[0]):
                if i < 0:
                    split_value = self.previous_x[i]
                    differences_from_samples_to_splitvalue = np.sum(np.square(np.subtract(self.previous_x , split_value)))


        self.thresholds_previous_samples = best_splits_previous_samples



class MultiDimensionalDecisionTree:
    def __init__(self,important_channels,movements, windom_size=150,num_trees=100,sample_difference_overlap=64,max_depth = 5,min_samples_split=20,num_previous_samples = 0):
        self.max_depth = max_depth # max depth of the tree
        self.min_samples_split = min_samples_split # min number of samples to split
        self.important_channels = important_channels # all the channels that are important between 0-320
        self.num_channels = len(self.important_channels) # number of channels that are important
        self.num_trees = num_trees # number of trees in the random forest
        self.movements = movements # all the movements that we want to classify
        self.window_size = windom_size # window size for the data ( length of the window in ms)
        self.window_size_in_samples = int((self.window_size / 1000) * 2048) # window size in samples
        self.sample_difference_overlap = sample_difference_overlap # difference between the start of the next window and the start of the previous window in samples
        self.overlap = self.window_size_in_samples - sample_difference_overlap # overlap between windows in samples
        self.movment_dict = {}
        self.num_previous_samples = num_previous_samples

        for i in range(len(self.movements)): #check if finger already in dict and add the movement number to the dict
            for part in ["index","thumb","middle","ring","pinky","fist","2pinch","3pinch"]:
                matching_keys = [key for key in self.movment_dict.keys() if part in key]
                if len(matching_keys) > 0:
                    self.movment_dict[self.movements[i]] = matching_keys[0]
                else:
                    self.movment_dict[self.movements[i]] = i

        self.build_training_data(self.movements, r"D:\Lab\data\extracted\Sub2")
        self.trees = []
        for i in range(self.num_trees):
            root = TreeNode(is_root=True,num_previous_samples=self.num_previous_samples)
            self.build_tree(root,self.X_train,self.y_train,self.max_depth,self.min_samples_split)
            self.trees.append(root)

    def display_tree(self,node:TreeNode, level=0):
        """Recursively display the tree structure with threshold values."""
        if node is None:
            return
        prefix = "  " * level  # Used for indentation
        print(prefix + f"Node {str(id(node))[-5:]}: Thresholds: {node.thresholds}")
        for child in node.children:
            self.display_tree(child, level + 1)

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
        for movement in tqdm.tqdm(movement_names,desc="Building training data"):
            emg_data, Mu_data, ref_data = open_all_files_for_one_patient_and_movement(path_to_subject_dat, movement)
            ref_erweitert = np.zeros((len(self.movements),len(ref_data))) # [num_movements x num_samples]
            ref_data = normalize_2D_array(ref_data)
            ref_erweitert[self.movment_dict[movement],:] = ref_data # jetzt werte für die bewegung an passenden index eintragen für anderen finger einträge auf 0.5 setzen
            emg_data = reshape_emg_channels_into_320(emg_data)
            for i in range(0, len(emg_data[0]) - window_size + 1, overlap): # da unterschiedliche länge von emg und ref nur machen wenn ref noch nicht zuzende ist
                if i <= ref_data.shape[1]:
                    segment = calculate_emg_rms_row(emg_data,i,self.window_size_in_samples)
                    # feature = calculate_rms(segment)
                    # segments.append(feature)
                    segments.append(segment)
                    labels.append(ref_erweitert[:,i])

        # Step 5: Combine segments and labels into your dataset
        # dataset = np.column_stack((np.array(segments), np.array(labels)))

        # Step 6: Train-Test Split (adjust split_ratio)
        split_index = int(len(segments) * split_ratio)
        self.train_data = segments[:split_index]
        self.train_valitdation = labels[split_index]
        self.test_data = segments[split_index:]
        self.test_validation = labels[split_index:]

        indices = np.random.permutation(len(self.test_data))

        self.shuffled_train_data = np.random.permutation(self.train_data)
        self.shuffled_test_data = np.array([self.test_data[i] for i in indices])
        self.shuffled_test_validation = np.array([self.test_validation[i] for i in indices])

    def add_data(self,x_train,y_train,x_test,y_test):
        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test

    def meets_criteria(self,entry, thresholds):
        # Return False if the entry is shorter than the thresholds list
        if len(entry) < len(thresholds):
            return False
        for i, threshold in enumerate(thresholds):
            if not entry[i] < threshold:
                return False
        return True

    def build_tree(self,node : TreeNode, X, y, depth=3, min_num_splits=2, num_previous_samples=0):

        if depth == 0 or y[0].shape[0] < min_num_splits:
            node.is_leaf = True
            node.thresholds= np.mean(y,axis=1) #this will then be the prediction value
            return

        node.find_best_splits(X)

        if node.is_leaf:
            return
        # TODO hier unterschiedung ob num_previous = 0, wenn ja dann wie unten wenn nein dann lef, und right mit anderen criterion testen ( nur die bei denen Zeitabstand kleiner als thershold ist)
        # split data and continue building the tree recursively
        left_y = [y[index] for index, entry in enumerate(y) if self.meets_criteria(entry, node.thresholds)]
        right_y = [y[index] for index, entry in enumerate(y) if not self.meets_criteria(entry, node.thresholds)]

        left_X = [entry for entry in X if self.meets_criteria(entry, node.thresholds)]
        right_X = [entry for entry in X if not self.meets_criteria(entry, node.thresholds)]

        left_child = node.add_child()
        self.build_tree(left_child, left_X,left_y, depth - 1, min_num_splits,num_previous_samples)

        right_child = node.add_child()
        self.build_tree(right_child, right_X,right_y, depth - 1, min_num_splits,num_previous_samples)

    # def fit(self, emg_values, force_values):
    #     # Update buffer
    #     self.emg_buffer.append(emg_values)
    #     self.label_buffer.append(force_values)
    #
    #     if len(self.emg_buffer) > self.buffer_size:
    #         self.emg_buffer.pop(0)
    #         self.label_buffer.pop(0)
    #
    #     self.tree = self._build_tree(np.array(self.emg_buffer).reshape(-1, 320),
    #                                  np.array(self.label_buffer).reshape(-1, 2)[:, 1])
    #
    # def predict(self, emg_values, tree=None):
    #     if tree is None:
    #         tree = self.tree
    #
    #     # Base case: if we are at a regressor
    #     if 'regressor' in tree:
    #         features = np.hstack([self.emg_buffer, [emg_values]])
    #         return tree['regressor'].predict([features])[0]
    #
    #     # Recursive case: go deeper into the tree
    #     normalized_value = self._normalize([emg_values])[0]
    #     if normalized_value[0] < self.threshold:
    #         return self.predict(emg_values, tree['left'])
    #     else:
    #         return self.predict(emg_values, tree['right'])

    def normalize_emg(self, emg_data):
        return normalize_2D_array(emg_data)

if __name__ == "__main__":
    movements = ["thumb_slow", "thumb_fast", "index_slow", "index_fast",]

    model = MultiDimensionalDecisionTree([15,16,45,98],movements,num_previous_samples=3)


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
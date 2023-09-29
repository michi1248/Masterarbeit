from hmmlearn import hmm
import pandas as pd
from exo_controller.helpers import *

class HMM_model:
    def __init__(self,n_states=13, covariance_type="full", n_iter=1000, path_to_subject_dat = r"D:\Lab\data\extracted\Sub2"):
        self.n_states = n_states  # Number of hidden states
        self.covariance_type = covariance_type
        self.n_iterations =n_iter  # Number of training iterations
        self.models= []
        for i in range (0,320):
            self.models.append(hmm.GaussianHMM(n_components=self.n_states, n_iter=1000, covariance_type=self.covariance_type))
        self.path_to_subject_dat = path_to_subject_dat
        self.movements = ["thumb_slow", "thumb_fast", "index_slow", "index_fast", "middle_slow", "middle_fast", "ring_slow", "ring_fast", "pinky_slow", "pinky_fast", "fist", "2pinch", "3pinch"]
        self.labels_dict = {"thumb_slow": 0, "thumb_fast": 1, "index_slow": 2, "index_fast": 3, "middle_slow": 4, "middle_fast": 5, "ring_slow": 6, "ring_fast": 7, "pinky_slow": 8, "pinky_fast": 9, "fist": 10, "2pinch": 11, "3pinch": 12}
        self.build_training_data(self.movements, self.path_to_subject_dat)

    def build_training_data(self, movement_names, path_to_subject_dat,window_size=int((150 / 1000) * 2048), overlap=int((150 / 1000) * 2048)-150,split_ratio=0.8):
        segments = []
        labels = []
        for movement in tqdm.tqdm(movement_names):
            emg_data, Mu_data, ref_data = open_all_files_for_one_patient_and_movement(path_to_subject_dat,movement)
            emg_data = reshape_emg_channels_into_320(emg_data)
            for i in range(0, len(emg_data[0]) - window_size + 1, overlap):
                    segment = emg_data[:,i:i + window_size]
                    #feature = calculate_rms(segment)
                    label = self.labels_dict[movement]
                    #segments.append(feature)
                    segments.append(segment)
                    labels.append(label)

        # Step 5: Combine segments and labels into your dataset
        #dataset = np.column_stack((np.array(segments), np.array(labels)))

        # Step 6: Train-Test Split (adjust split_ratio)
        split_index = int(len(segments) * split_ratio)
        self.train_data = segments[:split_index]
        self.train_valitdation = labels[:split_index]
        self.test_data = segments[split_index:]
        self.test_validation = labels[split_index:]

        indices = np.random.permutation(len(self.test_data))

        self.shuffled_train_data = np.random.permutation(self.train_data)
        self.shuffled_test_data = np.array([self.test_data[i] for i in indices])
        self.shuffled_test_validation = np.array([self.test_validation[i] for i in indices])


    def train(self, X,i):
        self.models[i].fit(X[:500])

    def predict(self, X,i):
        self.prediction_result = self.models[i].predict(X)
        real_result = self.shuffled_test_validation[:]
        print("Accuracy: ", np.mean(self.prediction_result == real_result))
        #print("Prediction: ", self.prediction_result)
        #print("Real: ", real_result)
        return self.prediction_result

if __name__ == "__main__":
    hmm_model = HMM_model()
    num_channels = 150
    for i in tqdm.tqdm(range(num_channels),desc= "Training HMM model"):
        hmm_model.train(hmm_model.shuffled_train_data[:,i,:],i)

    for i in tqdm.tqdm(range(num_channels),desc= "Inferencing HMM model"):
        hmm_model.predict(hmm_model.shuffled_test_data[:,i,:],i)
        states = pd.unique(hmm_model.prediction_result)
        #print(states)
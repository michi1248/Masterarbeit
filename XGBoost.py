from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tqdm
import time
from helpers import *
import xgboost as xgb

class XGBoost_model:
    def __init__(self):
        self.clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2,device='cuda')
        self.movements = ["thumb_slow", "thumb_fast", "index_slow", "index_fast", "middle_slow", "middle_fast",
                          "ring_slow", "ring_fast", "pinky_slow", "pinky_fast", "fist", "2pinch", "3pinch"]
        self.labels_dict = {"thumb_slow": 0, "thumb_fast": 1, "index_slow": 2, "index_fast": 3, "middle_slow": 4,
                            "middle_fast": 5, "ring_slow": 6, "ring_fast": 7, "pinky_slow": 8, "pinky_fast": 9,
                            "fist": 10, "2pinch": 11, "3pinch": 12}
        self.build_training_data(self.movements, r"D:\Lab\data\extracted\Sub2")


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
        self.X_train = segments[:split_index]
        self.y_train = labels[:split_index]
        self.X_test = segments[split_index:]
        self.y_test = labels[split_index:]

        indices_train = np.random.permutation(len(self.X_train))
        indices_test = np.random.permutation(len(self.X_test))

        self.X_train = [self.X_train[i] for i in indices_train]
        self.y_train = [self.y_train[i] for i in indices_train]
        self.X_test = [self.X_test[i] for i in indices_test]
        self.y_test = [self.y_test[i] for i in indices_test]


    def train(self):
        start = time.time()
        self.clf.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)])
        gpu_res = self.clf.evals_result()
        print("GPU Training Time: %s seconds" % (str(time.time() - start)))

    def predict(self):
        return self.clf.predict(self.X_test)

    def save_model(self):
        self.clf.save_model("clf.json")

if __name__ == "__main__":
    model = XGBoost_model()
    model.train()
    model.save_model()
    print(model.predict())




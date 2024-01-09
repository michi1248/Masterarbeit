import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from exo_controller import helpers
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.init as init






class ShallowConvNetWithAttention(nn.Module):
    def __init__(self, use_difference_heatmap=False, best_time_tree=0, grid_aranger=None,number_of_grids=2,use_channel_attention=False, use_spatial_attention=False):
        super(ShallowConvNetWithAttention, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store the attention flags
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention

        self.number_of_grids = number_of_grids
        self.use_difference_heatmap = use_difference_heatmap
        self.best_time_index = best_time_tree
        self.grid_aranger = grid_aranger

        # New convolutional layer to cover the entire grid
        self.grid_conv = nn.Conv2d(1, 2, kernel_size=(8, 8 * self.number_of_grids),padding=0)

        self.fc_overall1 = nn.Linear(8* 8 * self.number_of_grids,80)
        self.fc_overall2 = nn.Linear(80,2)
        self.fc_combined = nn.Linear(4,2)


        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)


        self.fc1 = nn.Linear(4096, 100)
        self.fc2 = nn.Linear(100, 2)
        self.relu = nn.ReLU()
        self.to(self.device)

    def _initialize_weights(self,m,seed=42):

        torch.manual_seed(seed)

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)


    def forward(self, heatmap1, heatmap2=None):
        if self.use_difference_heatmap:

            # activity = self.relu(self.grid_conv(heatmap1))



            # Process the first heatmap
            x1 = self.relu(self.conv1_heatmap1(heatmap1))
            x1 = self.relu(self.conv2_heatmap1(x1))

            # Process the second heatmap
            #x2 = self.relu(self.grid_conv(heatmap2))
            x2 = self.relu(self.conv1_heatmap2(heatmap2))
            x2 = self.relu(self.conv2_heatmap2(x2))


            # Combine the features from both heatmaps
            # Here we're using concatenation; you might choose a different method
            combined = torch.cat((x1, x2), dim=1)

            # Flatten and pass through fully connected layers
            combined = combined.view(combined.size(0), -1)
            combined = torch.relu(self.fc1(combined))
            combined = torch.relu(self.fc2(combined))



            return combined
        else:

            activity = torch.relu(self.fc_overall1(heatmap1.view(heatmap1.size(0), -1)))
            activity = torch.relu(self.fc_overall2(activity))

            #x = self.relu(self.grid_conv(heatmap1))
            x = self.relu(self.conv1(heatmap1))
            x = self.relu(self.conv2(x))
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))  # Sigmoid for values between 0 and 1

            combined = torch.cat((x, activity), dim=-1)
            combined = combined.view(combined.size(0), -1)
            combined = torch.relu(self.fc_combined(combined))

            return activity

    def train_model(self, train_loader, learning_rate=0.0001, epochs=10):
        self.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate,weight_decay=0.0001)

        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for inputs, targets in train_loader:


                inputs, targets = inputs.float(), targets.float()  #
                if self.use_difference_heatmap:
                    targets, _ = torch.unbind(targets, dim=0)
                    heatmap1, heatmap2 = torch.unbind(inputs, dim=0)
                    heatmap1 = heatmap1.view(-1, 1, 8, 8*self.number_of_grids)  # Reshape input
                    heatmap2 = heatmap2.view(-1, 1, 8, 8*self.number_of_grids)  # Reshape inpu

                    heatmap1 = heatmap1.to(self.device)
                    heatmap2 = heatmap2.to(self.device)
                else:
                    heatmap1 = inputs.view(-1, 1, 8, 8*self.number_of_grids)
                    heatmap1 = heatmap1.to(self.device)

                targets = targets.to(self.device)

                optimizer.zero_grad()
                if self.use_difference_heatmap:
                    outputs = self(heatmap1, heatmap2)
                else:
                    outputs = self(heatmap1)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': epoch_loss / ( len(train_loader))})

            progress_bar.close()

    def predict(self, heatmap1, heatmap2 = None):
        #TODO hier muss noch die grid aranger funktion rein
        with torch.no_grad():
            if not self.use_difference_heatmap:
                x = torch.from_numpy(heatmap1)
                x = x.float()  # Convert to float
                x = x.view(-1, 1, 8, 8*self.number_of_grids)  # Reshape input
                x = x.to(self.device)
                return self(x).cpu().numpy()
            else:
                x1 = torch.from_numpy(heatmap1)
                x1 = x1.float()
                x1 = x1.view(-1, 1, 8, 8*self.number_of_grids)
                x1 = x1.to(self.device)

                x2 = torch.from_numpy(heatmap2)
                x2 = x2.float()
                x2 = x2.view(-1, 1, 8, 8*self.number_of_grids)
                x2 = x2.to(self.device)
                return self(x1, x2).cpu().numpy()

    def evaluate_best(self,predictions,ground_truth):
        self.eval()

        if isinstance(predictions, list):
            predictions = torch.tensor(np.array(predictions), dtype=torch.float32)
        if isinstance(ground_truth, list):
            ground_truth = torch.tensor(np.array(ground_truth), dtype=torch.float32)

        r_squared_values = []
        for i in range(predictions.shape[1]):  # Assuming the second dimension contains the outputs
            sst = torch.sum((ground_truth[:, i] - torch.mean(ground_truth[:, i])) ** 2)
            ssr = torch.sum((ground_truth[:, i] - predictions[:, i]) ** 2)
            r_squared_values.append(1 - ssr / sst)

        crit = nn.MSELoss()
        # Separate the tensors into two parts
        ground_truth_0 = ground_truth[:, 0]
        ground_truth_1 = ground_truth[:, 1]

        prediction_0 = predictions[:, 0]
        prediction_1 = predictions[:, 1]

        # Calculate the loss for each part
        loss_0 = crit(ground_truth_0, prediction_0)
        loss_1 = crit(ground_truth_1, prediction_1)
        mse_loss = (loss_0+ loss_1)/2

        # Calculate the mean R-squared value
        mean_r_squared = torch.mean(torch.tensor(r_squared_values))
        print(f'Average Loss: {mse_loss}')
        return mean_r_squared.item(),mse_loss.item()


    def evaluate(self, test_loader):
        self.eval()
        total_loss = 0
        criterion = nn.MSELoss()
        all_targets = []
        all_predictions = []

        with torch.no_grad():

            for inputs, targets in test_loader:
                inputs, targets = inputs.float(), targets.float()  #
                if self.use_difference_heatmap:
                    targets,_ = torch.unbind(targets, dim=0)
                    heatmap1, heatmap2 = torch.unbind(inputs, dim=0)
                    heatmap1 = heatmap1.view(-1, 1, 8, 8*self.number_of_grids)  # Reshape input
                    heatmap2 = heatmap2.view(-1, 1, 8, 8*self.number_of_grids)  # Reshape inpu

                    heatmap1 = heatmap1.to(self.device)
                    heatmap2 = heatmap2.to(self.device)
                else:
                    heatmap1 = inputs.view(-1, 1, 8, 8*self.number_of_grids)
                    heatmap1 = heatmap1.to(self.device)

                targets = targets.to(self.device)
                if self.use_difference_heatmap:
                    outputs = self(heatmap1, heatmap2)
                else:
                    outputs = self(heatmap1)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())


        avg_loss = total_loss / len(test_loader)
        print(f'Average Loss: {avg_loss}')

        # Convert lists to numpy arrays for plotting
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)

        # Plotting
        plt.figure(figsize=(12, 6))

        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.scatter(np.arange(len(all_targets)), all_targets[:, i], color='blue', label='True Values')
            plt.scatter(np.arange(len(all_predictions)), all_predictions[:, i], color='red', label='Predictions')
            for j in range(len(all_targets)):
                plt.plot([j, j], [all_targets[j, i], all_predictions[j, i]], color='gray', linestyle='--')
            plt.title(f'Comparison of True Values and Predictions (Output {i + 1})')
            plt.legend()

        plt.show()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load_model(file_path):
        model = ShallowConvNetWithAttention()
        model.load_state_dict(torch.load(file_path))
        model.eval()
        return model

    def load_trainings_data(self,patient_number):
        X_test = np.array(
            helpers.load_pickle_file(
                r"trainings_data/resulting_trainings_data/subject_"
                + str(patient_number)
                + "/X_test_local.pkl"
            )
        )
        y_test = np.array(
            helpers.load_pickle_file(
                r"trainings_data/resulting_trainings_data/subject_"
                + str(patient_number)
                + "/y_test_local.pkl"
            )
        )
        X_train = np.array(
            helpers.load_pickle_file(
                r"trainings_data/resulting_trainings_data/subject_"
                + str(patient_number)
                + "/X_train_local.pkl"
            )
        )
        y_train = np.array(
            helpers.load_pickle_file(
                r"trainings_data/resulting_trainings_data/subject_"
                + str(patient_number)
                + "/y_train_local.pkl"
            )
        )

        X_train = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(X_train).transpose(2,1,0)
        X_test = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(X_test).transpose(2,1,0)
        # y_train = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(y_train).transpose(2,1,0)
        # y_test = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(y_test).transpose(2,1,0)



        if not self.use_difference_heatmap:
            # Create the data loaders
            train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
            train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
            self.train_loader = train_loader
            self.test_loader = test_loader
            return train_loader, test_loader

        else: # self.use_difference_heatmap:
            self.training_data_time = helpers.load_pickle_file(
                r"trainings_data/resulting_trainings_data/subject_"
                + str(patient_number)
                + "/training_data_time.pkl"
            )[self.best_time_index]
            #X_train, X_test, y_train, y_test is order in time data
            X_train_time = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(self.training_data_time[0]).transpose(2,1,0)
            X_test_time = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(self.training_data_time[1]).transpose(2,1,0)
            # y_train_time = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(self.training_data_time[2]).transpose(2,1,0)
            # y_test_time = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(self.training_data_time[3]).transpose(2,1,0)
            y_train_time = self.training_data_time[2]
            y_test_time = self.training_data_time[3]

            train_dataset_time = TensorDataset(torch.from_numpy(np.stack((X_train,X_train_time),axis=0)), torch.from_numpy(np.stack((y_train,y_train_time),axis=0)))
            test_dataset_time = TensorDataset(torch.from_numpy(np.stack((X_test,X_test_time),axis=0)), torch.from_numpy(np.stack((y_test,y_test_time),axis=0)))


            train_loader_time = DataLoader(train_dataset_time, batch_size=10, shuffle=False)
            test_loader_time = DataLoader(test_dataset_time, batch_size=10, shuffle=False)
            self.train_loader = train_loader_time
            self.test_loader  = test_loader_time
            return train_loader_time, test_loader_time





if __name__ == "__main__":
    # input : heatmap:  1 x 8 x 16 ( 1 channel, 8x16 größe)
    # conv layer über ganzes bild -> in channels = 1 , out channels = 2 , kernel size = 8x16  ( output hat shape 1,
    # -> zwei output weil ich ja zwei finger einzeln habe vielleicht lernt dann für jeden finger eine zahl
    #
    #
    #
    #
    #
    #
    pass
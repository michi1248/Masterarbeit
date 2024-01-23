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
    def __init__(self, use_difference_heatmap=False, best_time_tree=0, grid_aranger=None,number_of_grids=2,use_mean = None):
        super(ShallowConvNetWithAttention, self).__init__()
        self.use_mean = use_mean
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.number_of_grids = number_of_grids
        self.use_difference_heatmap = use_difference_heatmap
        self.best_time_index = best_time_tree
        self.grid_aranger = grid_aranger
        # Dropout rate
        self.dropout_rate = 0.2

        if self.use_difference_heatmap:
            # Global Activity Path
            self.global_pool1 = nn.AdaptiveAvgPool2d(1)
            self.global_pool2 = nn.AdaptiveAvgPool2d(1)

            # Spatial Activity Path
            self.conv1_1 = nn.Conv2d(number_of_grids, 64*number_of_grids, groups=number_of_grids, kernel_size=5, padding=1)
            self.pool_1 = nn.MaxPool2d(2, 2)
            self.conv2_1 = nn.Conv2d(64* number_of_grids, 32, kernel_size=5, padding=1)
            self.fc1_1 = nn.Linear(512, 60)   # channels per electrode * number of grids * input  (64 * 2 * 32)

            # Add Batch Normalization after convolutional layers
            self.bn1_1 = nn.BatchNorm2d(64*number_of_grids)
            self.bn2_1 = nn.BatchNorm2d(32)

            # Instance Normalization
            self.in1_1 = nn.InstanceNorm2d(64*number_of_grids)
            self.in2_1 = nn.InstanceNorm2d(32)
            self.fc2_1 = nn.Linear(60, 2)

            # Spatial Activity Path
            self.conv1_2 = nn.Conv2d(number_of_grids, 64*number_of_grids, groups=number_of_grids, kernel_size=5, padding=1)
            self.pool_2 = nn.MaxPool2d(2, 2)
            self.conv2_2 = nn.Conv2d(64* number_of_grids, 32, kernel_size=5, padding=1)
            self.fc1_2 = nn.Linear(512, 60)

            # Add Batch Normalization after convolutional layers
            self.bn1_2 = nn.BatchNorm2d(64*number_of_grids)
            self.bn2_2 = nn.BatchNorm2d(32)

            # Instance Normalization
            self.in1_2 = nn.InstanceNorm2d(64*number_of_grids)
            self.in2_2 = nn.InstanceNorm2d(32)
            self.fc2_2 = nn.Linear(60, 2)

            self.merge_layer = nn.Linear(2*number_of_grids+ 4, 2)


        else:
            # Global Activity Path
            self.global_pool = nn.AdaptiveAvgPool2d(1)

            # Spatial Activity Path
            self.conv1 = nn.Conv2d(number_of_grids, 64* number_of_grids, groups=number_of_grids, kernel_size=5, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(64* number_of_grids, 32, kernel_size=5, padding=1)
            self.fc1 = nn.Linear(512, 60)

            # Add Batch Normalization after convolutional layers
            self.bn1 = nn.BatchNorm2d(64*number_of_grids)
            self.bn2 = nn.BatchNorm2d(32)

            #Instance Normalization
            self.in1 = nn.InstanceNorm2d(64*number_of_grids)
            self.in2 = nn.InstanceNorm2d(32)
            self.fc2 = nn.Linear(60  ,2)
            self.merge_layer = nn.Linear(number_of_grids + 2 ,2)

        # Dropout rate
        self.dropout_rate = 0.2
        self.to(self.device)

    def _initialize_weights(self,m,seed=42):
        torch.manual_seed(seed)
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)


    def forward(self, heatmap1, heatmap2=None):

        if self.use_difference_heatmap:
            # Split the input into two parts
            if self.number_of_grids == 5:
                chunks = torch.chunk(heatmap1, 2, dim=2)
                good_chunks = chunks[0]
                bad_chunks = chunks[1]
                bad_chunks = torch.chunk(bad_chunks, 3, dim=3)
                good_chunks_split = torch.chunk(good_chunks, 3, dim=3)
                concat_tensors = list(good_chunks_split) + [bad_chunks[0], bad_chunks[1]]
                stacked_input1 = torch.cat(concat_tensors, dim=1)

                chunks = torch.chunk(heatmap2, 2, dim=2)
                good_chunks = chunks[0]
                bad_chunks = chunks[1]
                bad_chunks = torch.chunk(bad_chunks, 3, dim=3)
                good_chunks_split = torch.chunk(good_chunks, 3, dim=3)
                concat_tensors = list(good_chunks_split) + [bad_chunks[0], bad_chunks[1]]
                stacked_input2 = torch.cat(concat_tensors, dim=1)
            else:
                split_images1 = torch.chunk(heatmap1,self.number_of_grids,dim=3)
                stacked_input1 = torch.cat(split_images1, dim=1)
                split_images2 = torch.chunk(heatmap2,self.number_of_grids,dim=3)
                stacked_input2 = torch.cat(split_images2, dim=1)  # New shape will be [batch_size, 2, 8, 8]

            # Global Activity Path
            global_path1 = self.global_pool1(stacked_input1)
            global_path1 = global_path1.view(global_path1.size(0), -1)  # Flatten
            global_path1 = torch.square(global_path1)  # Average over the channels

            global_path2 = self.global_pool2(stacked_input2)
            global_path2 = global_path2.view(global_path2.size(0), -1)  # Flatten
            global_path2 = torch.square(global_path2)  # Average over the channels

            # Spatial Activity Path
            gelu = torch.nn.GELU(approximate='tanh')
            spatial_path1 = self.conv1_1(stacked_input1)
            spatial_path1 = self.bn1_1(spatial_path1)
            spatial_path1 = self.in1_1(spatial_path1)  # Uncomment if using instance normalization
            spatial_path1 = torch.nn.GELU(approximate='tanh')(spatial_path1)
            # spatial_path = self.pool(spatial_path)

            spatial_path1 = self.conv2_1(spatial_path1)
            spatial_path1 = self.bn2_1(spatial_path1)
            spatial_path1 = self.in2_1(spatial_path1)  # Uncomment if using instance normalization
            spatial_path1 = torch.nn.GELU(approximate='tanh')(spatial_path1)
            # spatial_path = self.pool(spatial_path)

            spatial_path1 = spatial_path1.view(spatial_path1.size(0), -1)
            spatial_path1 = F.dropout(spatial_path1, p=self.dropout_rate, training=self.training)  # Dropout after conv2
            spatial_path1 = self.fc1_1(spatial_path1)
            spatial_path1 = F.dropout(spatial_path1, p=self.dropout_rate, training=self.training)  # Dropout after fc1
            spatial_path1 = self.fc2_1(spatial_path1)

            # Spatial Activity Path
            gelu = torch.nn.GELU(approximate='tanh')
            spatial_path2 = self.conv1_2(stacked_input2)
            spatial_path2 = self.bn1_2(spatial_path2)
            spatial_path2 = self.in1_2(spatial_path2)  # Uncomment if using instance normalization
            spatial_path2 = torch.nn.GELU(approximate='tanh')(spatial_path2)
            # spatial_path = self.pool(spatial_path)

            spatial_path2 = self.conv2_2(spatial_path2)
            spatial_path2 = self.bn2_2(spatial_path2)
            spatial_path2 = self.in2_2(spatial_path2)  # Uncomment if using instance normalization
            spatial_path2 = torch.nn.GELU(approximate='tanh')(spatial_path2)
            # spatial_path = self.pool(spatial_path)

            spatial_path2 = spatial_path2.view(spatial_path2.size(0), -1)
            spatial_path2 = F.dropout(spatial_path2, p=self.dropout_rate, training=self.training)  # Dropout after conv2
            spatial_path2 = self.fc1_2(spatial_path2)
            spatial_path2 = F.dropout(spatial_path2, p=self.dropout_rate, training=self.training)  # Dropout after fc1
            spatial_path2 = self.fc2_2(spatial_path2)

            merged_spatial_path = torch.cat((spatial_path1, spatial_path2), dim=1)


            global_path1 = global_path1.view(global_path1.size(0), -1)
            global_path2 = global_path2.view(global_path2.size(0), -1)
            merged = torch.cat((merged_spatial_path,global_path1, global_path2), dim=1)
            merged_spatial_path = self.merge_layer(merged)
            return merged_spatial_path

        else:

            if self.number_of_grids == 5:
                chunks = torch.chunk(heatmap1, 2, dim=2)
                good_chunks = chunks[0]
                bad_chunks = chunks[1]
                bad_chunks = torch.chunk(bad_chunks, 3, dim=3)

                good_chunks_split = torch.chunk(good_chunks, 3, dim=3)
                concat_tensors = list(good_chunks_split) + [bad_chunks[0], bad_chunks[1]]
                stacked_input = torch.cat(concat_tensors, dim=1)
            else:
                # Split the input into two parts
                split_images = torch.chunk(heatmap1, self.number_of_grids, dim=3)  # This will create two tensors of shape [batch_size, 1, 8, 8]
                stacked_input = torch.cat(split_images, dim=1)  # New shape will be [batch_size, 2, 8, 8]

            # Global Activity Path
            global_path = self.global_pool(stacked_input)
            global_path = global_path.view(global_path.size(0), -1)  # Flatten
            global_path = torch.square(global_path)  # Average over the channels

            # Spatial Activity Path
            gelu = torch.nn.GELU(approximate='tanh')
            spatial_path = self.conv1(stacked_input)
            spatial_path = self.bn1(spatial_path)
            spatial_path = self.in1(spatial_path)  # Uncomment if using instance normalization
            spatial_path = torch.nn.GELU(approximate='tanh')(spatial_path)
            #spatial_path = self.pool(spatial_path)

            spatial_path = self.conv2(spatial_path)
            spatial_path = self.bn2(spatial_path)
            spatial_path = self.in2(spatial_path)  # Uncomment if using instance normalization
            spatial_path = torch.nn.GELU(approximate='tanh')(spatial_path)
            #spatial_path = self.pool(spatial_path)

            spatial_path = spatial_path.view(spatial_path.size(0), -1)
            spatial_path = F.dropout(spatial_path, p=self.dropout_rate, training=self.training)  # Dropout after conv2
            spatial_path = self.fc1(spatial_path)
            spatial_path = F.dropout(spatial_path, p=self.dropout_rate, training=self.training)  # Dropout after fc1
            spatial_path = self.fc2(spatial_path)

            gobal_path = global_path.view(global_path.size(0), -1)
            merged_spatial_path = torch.cat((spatial_path, gobal_path), dim=1)
            merged_spatial_path = self.merge_layer(merged_spatial_path)


            return merged_spatial_path

    def train_model(self, train_loader, learning_rate=0.00001, epochs=10):
        self.train()
        criterion = nn.L1Loss()
        #criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate,weight_decay=0.1,amsgrad=True)
        #early_stopping = EarlyStopping(patience=5, min_delta=0.001)  # Adjust as needed

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * (10 ** 1.5),
            total_steps=train_loader.__len__()*epochs,
            anneal_strategy="cos",
            three_phase=False,
            div_factor=10 ** 1.5,
            final_div_factor=1e3,
        )
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)

        progress_bar = tqdm(train_loader, desc="Epochs", leave=False, total=epochs)
        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.float(), targets.float()  #
                if self.use_difference_heatmap:
                    targets, _ = torch.unbind(targets, dim=0)
                    heatmap1, heatmap2 = torch.unbind(inputs, dim=0)
                    if self.number_of_grids == 5:
                        heatmap1 = heatmap1.view(-1, 1, 16,24)  # Reshape input
                        heatmap2 = heatmap2.view(-1, 1, 16, 24)  # Reshape inpu
                    else:
                        heatmap1 = heatmap1.view(-1, 1, 8, 8*self.number_of_grids)  # Reshape input
                        heatmap2 = heatmap2.view(-1, 1, 8, 8*self.number_of_grids)  # Reshape inpu

                    heatmap1 = heatmap1.to(self.device)
                    heatmap2 = heatmap2.to(self.device)
                else:
                    if self.number_of_grids == 5:
                        heatmap1 = inputs.view(-1, 1, 16, 24)  # Reshape input
                    else:
                        heatmap1 = inputs.view(-1, 1, 8, 8*self.number_of_grids)
                    heatmap1 = heatmap1.to(self.device)
                targets = targets.to(self.device)

                if self.use_difference_heatmap:
                    output  = self(heatmap1, heatmap2)
                else:
                    output = self(heatmap1)

                output1 = output[:,0]
                output2 = output[:,1]
                loss1 = criterion(output1, targets[:,0])
                loss2 = criterion(output2, targets[:,1])
                total_loss = (loss1 + loss2) #* 100

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

                scheduler.step()

            progress_bar.update(1)

        progress_bar.close()

    def predict(self, heatmap1, heatmap2 = None):
        #TODO hier muss noch die grid aranger funktion rein
        with torch.no_grad():
            if not self.use_difference_heatmap:
                x = torch.from_numpy(heatmap1)
                x = x.float()  # Convert to float
                if self.number_of_grids == 5:
                    x = x.view(-1, 1, 16, 24)
                else:
                    x = x.view(-1, 1, 8, 8*self.number_of_grids)  # Reshape input
                x = x.to(self.device)
                # pred  = self(x).cpu().numpy()
                # if pred[0] < 0:
                #     pred[0] = 0
                # if pred[1] < 0:
                #     pred[1] = 0
                # if pred[0] > 1:
                #     pred[0] = 1
                # if pred[1] > 1:
                #     pred[1] = 1
                # return pred
                return self(x).cpu().numpy()
            else:
                x1 = torch.from_numpy(heatmap1)
                x1 = x1.float()
                if self.number_of_grids == 5:
                    x1 = x1.view(-1, 1, 16, 24)
                    x1 = x1.to(self.device)

                    x2 = torch.from_numpy(heatmap2)
                    x2 = x2.float()
                    x2 = x2.view(-1, 1, 16,24)
                    x2 = x2.to(self.device)
                else:
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
            r_squared_values.append(1 - ssr / (sst + 1e-8))

        #crit = nn.MSELoss()
        crit = nn.L1Loss()

        # Separate the tensors into two parts
        ground_truth_0 = ground_truth[:, 0]
        ground_truth_1 = ground_truth[:, 1]

        prediction_0 = predictions[:, 0]
        prediction_1 = predictions[:, 1]

        # Calculate the loss for each part
        loss_0 = crit(ground_truth_0, prediction_0)
        loss_1 = crit(ground_truth_1, prediction_1)
        mse_loss = (loss_0+ loss_1)/(2)

        # Calculate the mean R-squared value
        mean_r_squared = torch.mean(torch.tensor(r_squared_values))
        print(f'Average Loss to next recording: {mse_loss}')
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
                    if self.number_of_grids == 5:
                        heatmap1 = heatmap1.view(-1, 1, 16,24)
                        heatmap2 = heatmap2.view(-1, 1, 16, 24)
                    else:
                        heatmap1 = heatmap1.view(-1, 1, 8, 8*self.number_of_grids)  # Reshape input
                        heatmap2 = heatmap2.view(-1, 1, 8, 8*self.number_of_grids)  # Reshape inpu

                    heatmap1 = heatmap1.to(self.device)
                    heatmap2 = heatmap2.to(self.device)
                else:
                    if self.number_of_grids == 5:
                        heatmap1 = inputs.view(-1, 1, 16, 24)
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
        print(f'Average Loss Test Data: {avg_loss}')

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

    @classmethod
    def load_and_further_train(cls, file_path, train_loader, additional_epochs=10, new_learning_rate=0.000001):
        """
        Load a pre-trained model and further train it with given data.

        :param file_path: Path to the pre-trained model file.
        :param train_loader: DataLoader for the training data.
        :param additional_epochs: Number of additional epochs to train.
        :param new_learning_rate: Learning rate for further training.
        :return: Trained model.
        """
        model = cls.load_model(file_path)
        model.train_model(train_loader, learning_rate=new_learning_rate, epochs=additional_epochs)
        return model

    def load_trainings_data(self,patient_number):
        if not self.use_mean :
            print("not using mean in Shallow Conv")
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
        else:
            print("Using mean in Shallow Conv")
            X_test = np.array(
                helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/X_test_local_mean.pkl"
                )
            )
            y_test = np.array(
                helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/y_test_local_mean.pkl"
                )
            )
            X_train = np.array(
                helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/X_train_local_mean.pkl"
                )
            )
            y_train = np.array(
                helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/y_train_local_mean.pkl"
                )
            )

        X_train = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(X_train).transpose(2,0,1)
        X_test = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(X_test).transpose(2,0,1)
        # y_train = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(y_train).transpose(2,1,0)
        # y_test = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(y_test).transpose(2,1,0)



        if not self.use_difference_heatmap:
            # Create the data loaders
            train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
            self.train_loader = train_loader
            self.test_loader = test_loader
            return train_loader, test_loader

        else: # self.use_difference_heatmap:
            if not self.use_mean:
                self.training_data_time = helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/training_data_time.pkl"
                )[self.best_time_index]
            else:
                self.training_data_time = helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/training_data_time_mean.pkl"
                )[self.best_time_index]
            #X_train, X_test, y_train, y_test is order in time data
            X_train_time = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(self.training_data_time[0]).transpose(2,0,1)
            X_test_time = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(self.training_data_time[1]).transpose(2,0,1)
            # y_train_time = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(self.training_data_time[2]).transpose(2,1,0)
            # y_test_time = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(self.training_data_time[3]).transpose(2,1,0)
            y_train_time = self.training_data_time[2]
            y_test_time = self.training_data_time[3]

            train_dataset_time = TensorDataset(torch.from_numpy(np.stack((X_train,X_train_time),axis=0)), torch.from_numpy(np.stack((y_train,y_train_time),axis=0)))
            test_dataset_time = TensorDataset(torch.from_numpy(np.stack((X_test,X_test_time),axis=0)), torch.from_numpy(np.stack((y_test,y_test_time),axis=0)))


            train_loader_time = DataLoader(train_dataset_time, batch_size=64, shuffle=True)
            test_loader_time = DataLoader(test_dataset_time, batch_size=64, shuffle=True)
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
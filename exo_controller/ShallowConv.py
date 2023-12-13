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

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # Adaptive Pooling to squeeze spatial dimensions
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Fully connected layers to learn channel-wise attention
        self.fc1 = nn.Linear(num_channels, num_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_channels // reduction_ratio, num_channels, bias=False)

    def forward(self, x):
        # Squeeze using average pooling and max pooling
        avg_pooled = self.avg_pool(x).view(x.size(0), -1)
        max_pooled = self.max_pool(x).view(x.size(0), -1)

        # Apply fully connected layers to each pooled feature map
        avg_attention = self.fc2(self.relu(self.fc1(avg_pooled)))
        max_attention = self.fc2(self.relu(self.fc1(max_pooled)))

        # Combine the attentions and apply sigmoid activation
        attention = torch.sigmoid(avg_attention + max_attention).unsqueeze(2).unsqueeze(3)

        # Apply the attention to the input
        return x * attention.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, use_difference_heatmap=False):
        super(SpatialAttention, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        return x

class ShallowConvNetWithAttention(nn.Module):
    def __init__(self, use_difference_heatmap=False, best_time_tree=0, grid_aranger=None):
        super(ShallowConvNetWithAttention, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_difference_heatmap = use_difference_heatmap
        self.best_time_index = best_time_tree
        self.grid_aranger = grid_aranger

        if self.use_difference_heatmap:
            # Convolutional layers for the first heatmap
            self.conv1_heatmap1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2_heatmap1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.attention_heatmap1 = SpatialAttention().to(self.device)
            self.channel_attention1 = ChannelAttention(num_channels=32)  # Add the ChannelAttention layer

            # Convolutional layers for the second heatmap
            self.conv1_heatmap2 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2_heatmap2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.attention_heatmap2 = SpatialAttention().to(self.device)
            self.channel_attention2 = ChannelAttention(num_channels=32)  # Add the ChannelAttention layer

            # Fully connected layers
            # Adjust the input size based on how you combine the features
            self.fc1 = nn.Linear(32 * 8 * 24 * 2, 100)  # Assuming concatenation
            self.fc2 = nn.Linear(100, 2)

            self.relu = nn.ReLU()
            self.to(self.device)

        else:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.attention = SpatialAttention().to(self.device)
            self.channel_attention = ChannelAttention(num_channels=32)  # Add the ChannelAttention layer
            self.fc1 = nn.Linear(32 * 8 * 24, 100)
            self.fc2 = nn.Linear(100, 2)
            self.relu = nn.ReLU()
            self.to(self.device)

    def _initialize_weights(self,m):

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)


    def forward(self, heatmap1, heatmap2=None):
        if self.use_difference_heatmap:
            # Process the first heatmap
            x1 = self.attention_heatmap1(heatmap1) * heatmap1  # Apply attention
            x1 = self.relu(self.conv1_heatmap1(x1))
            x1 = self.relu(self.conv2_heatmap1(x1))
            x1 = self.channel_attention1(x1)  # Apply channel attention


            # Process the second heatmap
            x2 = self.attention_heatmap2(heatmap2) * heatmap2  # Apply attention
            x2 = self.relu(self.conv1_heatmap2(x2))
            x2 = self.relu(self.conv2_heatmap2(x2))
            x2 = self.channel_attention1(x2)  # Apply channel attention


            # Combine the features from both heatmaps
            # Here we're using concatenation; you might choose a different method
            combined = torch.cat((x1, x2), dim=1)

            # Flatten and pass through fully connected layers
            combined = combined.view(combined.size(0), -1)
            combined = self.relu(self.fc1(combined))
            combined = torch.sigmoid(self.fc2(combined))

            return combined
        else:
            x = self.attention(heatmap1) * heatmap1  # Apply attention
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.channel_attention(x)  # Apply channel attention
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = self.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))  # Sigmoid for values between 0 and 1
            return x

    def train_model(self, train_loader, learning_rate=0.0001, epochs=10):
        self.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for inputs, targets in train_loader:


                inputs, targets = inputs.float(), targets.float()  #
                if self.use_difference_heatmap:
                    targets, _ = torch.unbind(targets, dim=0)
                    heatmap1, heatmap2 = torch.unbind(inputs, dim=0)
                    heatmap1 = heatmap1.view(-1, 1, 8, 24)  # Reshape input
                    heatmap2 = heatmap2.view(-1, 1, 8, 24)  # Reshape inpu

                    heatmap1 = heatmap1.to(self.device)
                    heatmap2 = heatmap2.to(self.device)
                else:
                    heatmap1 = inputs.view(-1, 1, 8, 24)
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
                progress_bar.set_postfix({'loss': epoch_loss / len(train_loader)})

            progress_bar.close()

    def predict(self, heatmap1, heatmap2 = None):
        with torch.no_grad():
            if not self.use_difference_heatmap:
                x = torch.from_numpy(heatmap1)
                x = x.float()  # Convert to float
                x = x.view(-1, 1, 8, 24)  # Reshape input
                x = x.to(self.device)
                return self(x).cpu().numpy()
            else:
                x1 = torch.from_numpy(heatmap1)
                x1 = x1.float()
                x1 = x1.view(-1, 1, 8, 24)
                x1 = x1.to(self.device)

                x2 = torch.from_numpy(heatmap2)
                x2 = x2.float()
                x2 = x2.view(-1, 1, 8, 24)
                x2 = x2.to(self.device)
                return self(x1, x2).cpu().numpy()


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
                    heatmap1 = heatmap1.view(-1, 1, 8, 24)  # Reshape input
                    heatmap2 = heatmap2.view(-1, 1, 8, 24)  # Reshape inpu

                    heatmap1 = heatmap1.to(self.device)
                    heatmap2 = heatmap2.to(self.device)
                else:
                    heatmap1 = inputs.view(-1, 1, 8, 24)
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
            train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
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


            train_loader_time = DataLoader(train_dataset_time, batch_size=20, shuffle=False)
            test_loader_time = DataLoader(test_dataset_time, batch_size=20, shuffle=False)
            self.train_loader = train_loader_time
            self.test_loader  = test_loader_time
            return train_loader_time, test_loader_time





if __name__ == "__main__":
    # Load the data
    X_train = torch.load('data/X_train.pt')
    y_train = torch.load('data/y_train.pt')
    X_test = torch.load('data/X_test.pt')
    y_test = torch.load('data/y_test.pt')

    # Create the data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Create the model
    model = ShallowConvNetWithAttention()

    # Train the model
    model.train_model(train_loader, epochs=10)

    # Evaluate the model
    avg_loss = model.evaluate(test_loader)
    print('Average test loss: {:.4f}'.format(avg_loss))

    # Save the model
    model.save_model("shallow_conv_net_with_attention.pth")
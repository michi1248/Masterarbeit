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
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        return x

class ShallowConvNetWithAttention(nn.Module):
    def __init__(self):
        super(ShallowConvNetWithAttention, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.attention = SpatialAttention().to(self.device)
        self.fc1 = nn.Linear(32 * 8 * 24, 100)
        self.fc2 = nn.Linear(100, 2)
        self.relu = nn.ReLU()
        self.to(self.device)


    def _initialize_weights(self,m):

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.attention(x) * x  # Apply attention
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for values between 0 and 1
        return x

    def train_model(self, train_loader, learning_rate=0.001, epochs=10):
        self.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for inputs, targets in train_loader:
                inputs, targets = inputs.float(), targets.float()  #
                inputs = inputs.view(-1, 1, 8, 24)  # Reshape input
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': epoch_loss / len(train_loader)})

            progress_bar.close()

    def predict(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x)
            x = x.float()  # Convert to float
            x = x.view(-1, 1, 8, 24)  # Reshape input
            x = x.to(self.device)
            return self(x).cpu().numpy()

    def evaluate(self, test_loader):
        self.eval()
        total_loss = 0
        criterion = nn.MSELoss()
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.float(), targets.float()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.view(-1, 1, 8, 24)
                outputs = self(inputs)
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

        # Create the data loaders
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        train_loader = DataLoader(train_dataset, batch_size=30, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)
        self.train_loader = train_loader
        self.test_loader = test_loader
        return train_loader, test_loader



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
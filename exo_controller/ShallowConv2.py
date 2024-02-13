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
    def __init__(
        self,
        use_difference_heatmap=False,
        best_time_tree=0,
        grid_aranger=None,
        number_of_grids=2,
        use_mean=None,
        retrain=False,
        retrain_number=None,
        finger_indexes=None,
        use_muovi_pro=False,
    ):
        super(ShallowConvNetWithAttention, self).__init__()
        self.use_mean = use_mean
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using : ", self.device)

        self.finger_indexes = finger_indexes
        self.number_of_grids = number_of_grids
        self.use_difference_heatmap = use_difference_heatmap
        self.best_time_index = best_time_tree
        self.grid_aranger = grid_aranger
        # Dropout rate
        self.dropout_rate = 0.1
        self.retrain = retrain
        self.retrain_number = retrain_number
        self.batch_size = 128
        self.use_muovi_pro = use_muovi_pro
        # self.criterion = nn.HuberLoss()
        self.criterion = nn.L1Loss()
        # self.criterion = nn.MSELoss()

        if self.use_difference_heatmap:
            # Global Activity Path
            self.global_pool1 = nn.AdaptiveAvgPool2d(1)
            self.global_pool2 = nn.AdaptiveAvgPool2d(1)

            # Spatial Activity Path
            self.conv1_1 = nn.Conv2d(
                number_of_grids,
                64 * number_of_grids,
                groups=number_of_grids,
                kernel_size=5,
                padding=1,
            )
            self.pool_1 = nn.MaxPool2d(2, 2)
            self.conv2_1 = nn.Conv2d(64 * number_of_grids, 32, kernel_size=5, padding=1)
            self.fc1_1 = nn.Linear(
                512, 60
            )  # channels per electrode * number of grids * input  (64 * 2 * 32)

            # Add Batch Normalization after convolutional layers
            self.bn1_1 = nn.BatchNorm2d(64 * number_of_grids)
            self.bn2_1 = nn.BatchNorm2d(32)

            # Instance Normalization
            self.in1_1 = nn.InstanceNorm2d(64 * number_of_grids)
            self.in2_1 = nn.InstanceNorm2d(32)
            self.fc2_1 = nn.Linear(60, len(self.finger_indexes))

            # Spatial Activity Path
            self.conv1_2 = nn.Conv2d(
                number_of_grids,
                64 * number_of_grids,
                groups=number_of_grids,
                kernel_size=5,
                padding=1,
            )
            self.pool_2 = nn.MaxPool2d(2, 2)
            self.conv2_2 = nn.Conv2d(64 * number_of_grids, 32, kernel_size=5, padding=1)
            self.fc1_2 = nn.Linear(512, 60)

            # Add Batch Normalization after convolutional layers
            self.bn1_2 = nn.BatchNorm2d(64 * number_of_grids)
            self.bn2_2 = nn.BatchNorm2d(32)

            # Instance Normalization
            self.in1_2 = nn.InstanceNorm2d(64 * number_of_grids)
            self.in2_2 = nn.InstanceNorm2d(32)
            self.fc2_2 = nn.Linear(60, len(self.finger_indexes))

            self.merge_layer = nn.Linear(
                2 * number_of_grids + 2 * len(self.finger_indexes),
                len(self.finger_indexes),
            )

        else:
            # Global Activity Path
            self.global_pool = nn.AdaptiveAvgPool2d(1)

            if not self.use_muovi_pro:
                # Spatial Activity Path
                self.conv1 = nn.Conv2d(1, 70, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(70)
                self.in1 = nn.InstanceNorm2d(70)
                self.pool_1 = nn.MaxPool2d(2, 2)

                self.conv2 = nn.Conv2d(70, 70, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(70)
                self.in2 = nn.InstanceNorm2d(70)
                self.pool2 = nn.MaxPool2d(2, 2)

                self.dropout = nn.Dropout(0.1)

                self.conv3 = nn.Conv2d(70, 150, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm2d(150)
                self.in3 = nn.InstanceNorm2d(150)
                self.pool3 = nn.MaxPool2d(2, 2)

                self.conv4 = nn.Conv2d(150, 150, kernel_size=3, padding=1)
                self.bn4 = nn.BatchNorm2d(150)
                self.in4 = nn.InstanceNorm2d(150)
                self.pool4 = nn.MaxPool2d(2, 2)

                self.conv5 = nn.Conv2d(150, 75, kernel_size=1)
                self.bn5 = nn.BatchNorm2d(75)
                self.in5 = nn.InstanceNorm2d(75)
                self.pool5 = nn.MaxPool2d(2, 2)

                self.fc1 = nn.Linear(28800 , 500)
                self.fc2 = nn.Linear(500, len(self.finger_indexes))
                self.merge_layer = nn.Linear(
                    len(self.finger_indexes) + number_of_grids, len(self.finger_indexes)
                )
            else:

                # Spatial Activity Path
                self.conv1 = nn.Conv2d(1, 70, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(70)
                self.in1 = nn.InstanceNorm2d(70)
                self.pool_1 = nn.MaxPool2d(2, 2)

                self.conv2 = nn.Conv2d(70, 70, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(70)
                self.in2 = nn.InstanceNorm2d(70)
                self.pool2 = nn.MaxPool2d(2, 2)

                self.dropout = nn.Dropout(0.1)

                self.conv3 = nn.Conv2d(70, 150, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm2d(150)
                self.in3 = nn.InstanceNorm2d(150)
                self.pool3 = nn.MaxPool2d(2, 2)

                self.conv4 = nn.Conv2d(150, 150, kernel_size=3, padding=1)
                self.bn4 = nn.BatchNorm2d(150)
                self.in4 = nn.InstanceNorm2d(150)
                self.pool4 = nn.MaxPool2d(2, 2)

                self.conv5 = nn.Conv2d(150, 75, kernel_size=1)
                self.bn5 = nn.BatchNorm2d(75)
                self.in5 = nn.InstanceNorm2d(75)
                self.pool5 = nn.MaxPool2d(2, 2)

                self.fc1 = nn.Linear(2400, 500)
                self.fc2 = nn.Linear(500, len(self.finger_indexes))
                self.merge_layer = nn.Linear(
                    len(self.finger_indexes) + 1, len(self.finger_indexes)
                )




        # Dropout rate
        self.to(self.device)

    def _initialize_weights(self, m, seed=42):
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
                concat_tensors = list(good_chunks_split) + [
                    bad_chunks[0],
                    bad_chunks[1],
                ]
                stacked_input1 = torch.cat(concat_tensors, dim=1)

                chunks = torch.chunk(heatmap2, 2, dim=2)
                good_chunks = chunks[0]
                bad_chunks = chunks[1]
                bad_chunks = torch.chunk(bad_chunks, 3, dim=3)
                good_chunks_split = torch.chunk(good_chunks, 3, dim=3)
                concat_tensors = list(good_chunks_split) + [
                    bad_chunks[0],
                    bad_chunks[1],
                ]
                stacked_input2 = torch.cat(concat_tensors, dim=1)
            else:
                split_images1 = torch.chunk(heatmap1, self.number_of_grids, dim=3)
                stacked_input1 = torch.cat(split_images1, dim=1)
                split_images2 = torch.chunk(heatmap2, self.number_of_grids, dim=3)
                stacked_input2 = torch.cat(
                    split_images2, dim=1
                )  # New shape will be [batch_size, 2, 8, 8]

            # Global Activity Path
            global_path1 = self.global_pool1(stacked_input1)
            global_path1 = global_path1.view(global_path1.size(0), -1)  # Flatten
            global_path1 = torch.multiply(
                global_path1, torch.abs(global_path1)
            )  # Average over the channels

            global_path2 = self.global_pool2(stacked_input2)
            global_path2 = global_path2.view(global_path2.size(0), -1)  # Flatten
            global_path2 = torch.multiply(
                global_path2, torch.abs(global_path2)
            )  # Average over the channels

            # Spatial Activity Path
            spatial_path1 = self.conv1_1(stacked_input1)
            spatial_path1 = self.bn1_1(spatial_path1)
            spatial_path1 = self.in1_1(
                spatial_path1
            )  # Uncomment if using instance normalization
            spatial_path1 = torch.nn.GELU(approximate="tanh")(spatial_path1)
            # spatial_path = self.pool(spatial_path)

            spatial_path1 = self.conv2_1(spatial_path1)
            spatial_path1 = self.bn2_1(spatial_path1)
            spatial_path1 = self.in2_1(
                spatial_path1
            )  # Uncomment if using instance normalization
            spatial_path1 = torch.nn.GELU(approximate="tanh")(spatial_path1)
            # spatial_path = self.pool(spatial_path)

            spatial_path1 = spatial_path1.view(spatial_path1.size(0), -1)
            spatial_path1 = F.dropout(
                spatial_path1, p=self.dropout_rate, training=self.training
            )  # Dropout after conv2
            spatial_path1 = self.fc1_1(spatial_path1)
            spatial_path1 = F.dropout(
                spatial_path1, p=self.dropout_rate, training=self.training
            )  # Dropout after fc1
            spatial_path1 = self.fc2_1(spatial_path1)

            # Spatial Activity Path
            gelu = torch.nn.GELU(approximate="tanh")
            spatial_path2 = self.conv1_2(stacked_input2)
            spatial_path2 = self.bn1_2(spatial_path2)
            spatial_path2 = self.in1_2(
                spatial_path2
            )  # Uncomment if using instance normalization
            spatial_path2 = torch.nn.GELU(approximate="tanh")(spatial_path2)
            # spatial_path = self.pool(spatial_path)

            spatial_path2 = self.conv2_2(spatial_path2)
            spatial_path2 = self.bn2_2(spatial_path2)
            spatial_path2 = self.in2_2(
                spatial_path2
            )  # Uncomment if using instance normalization
            spatial_path2 = torch.nn.GELU(approximate="tanh")(spatial_path2)
            # spatial_path = self.pool(spatial_path)

            spatial_path2 = spatial_path2.view(spatial_path2.size(0), -1)
            spatial_path2 = F.dropout(
                spatial_path2, p=self.dropout_rate, training=self.training
            )  # Dropout after conv2
            spatial_path2 = self.fc1_2(spatial_path2)
            spatial_path2 = F.dropout(
                spatial_path2, p=self.dropout_rate, training=self.training
            )  # Dropout after fc1
            spatial_path2 = self.fc2_2(spatial_path2)

            merged_spatial_path = torch.cat((spatial_path1, spatial_path2), dim=1)

            global_path1 = global_path1.view(global_path1.size(0), -1)
            global_path2 = global_path2.view(global_path2.size(0), -1)
            merged = torch.cat((merged_spatial_path, global_path1, global_path2), dim=1)
            merged_spatial_path = self.merge_layer(merged)
            return merged_spatial_path

        else:
            if not self.use_muovi_pro:
                if self.number_of_grids == 5:
                    chunks = torch.chunk(heatmap1, 2, dim=2)
                    good_chunks = chunks[0]
                    bad_chunks = chunks[1]
                    bad_chunks = torch.chunk(bad_chunks, 3, dim=3)

                    good_chunks_split = torch.chunk(good_chunks, 3, dim=3)
                    concat_tensors = list(good_chunks_split) + [
                        bad_chunks[0],
                        bad_chunks[1],
                    ]
                    stacked_input = torch.cat(concat_tensors, dim=1)
                else:
                    # Split the input into two parts
                    split_images = torch.chunk(
                        heatmap1, self.number_of_grids, dim=3
                    )  # This will create two tensors of shape [batch_size, 1, 8, 8]
                    stacked_input = torch.cat(
                        split_images, dim=1
                    )  # New shape will be [batch_size, 2, 8, 8]

                # Global Activity Path

                global_path = self.global_pool(stacked_input)
                global_path = global_path.view(global_path.size(0), -1)  # Flatten
                global_path = torch.multiply(
                    global_path, torch.abs(global_path)
                )  # Average over the channels

                # Spatial Activity Path
                spatial_path = self.conv1(heatmap1)
                spatial_path = self.bn1(spatial_path)
                spatial_path = self.in1(
                    spatial_path
                )  # Uncomment if using instance normalization
                spatial_path = torch.nn.GELU(approximate="tanh")(spatial_path)
                # spatial_path = self.pool(spatial_path)

                spatial_path = self.conv2(spatial_path)
                spatial_path = self.bn2(spatial_path)
                spatial_path = self.in2(
                    spatial_path
                )  # Uncomment if using instance normalization
                spatial_path = torch.nn.GELU(approximate="tanh")(spatial_path)
                # spatial_path = self.pool(spatial_path)

                spatial_path = self.conv3(spatial_path)
                spatial_path = self.bn3(spatial_path)
                spatial_path = self.in3(
                    spatial_path
                )
                spatial_path = torch.nn.GELU(approximate="tanh")(spatial_path)

                spatial_path = self.conv4(spatial_path)
                spatial_path = self.bn4(spatial_path)
                spatial_path = self.in4(
                    spatial_path
                )
                spatial_path = torch.nn.GELU(approximate="tanh")(spatial_path)

                spatial_path = self.conv5(spatial_path)
                spatial_path = self.bn5(spatial_path)
                spatial_path = self.in5(
                    spatial_path
                )
                spatial_path = torch.nn.GELU(approximate="tanh")(spatial_path)

                spatial_path = spatial_path.view(spatial_path.size(0), -1)

                spatial_path = F.dropout(spatial_path, p=self.dropout_rate, training=self.training)  # Dropout
                spatial_path = self.fc1(spatial_path)
                spatial_path = F.dropout(spatial_path, p=self.dropout_rate, training=self.training)
                spatial_path = self.fc2(spatial_path)

                # heatmap1_only_channels = F.dropout(heatmap1_only_channels, p=self.dropout_rate, training=self.training)  # Dropout
                # channel_wise_spatial_path = self.fc_all1(heatmap1_only_channels)

                gobal_path = global_path.view(global_path.size(0), -1)
                # channel_wise_spatial_path = channel_wise_spatial_path.view(channel_wise_spatial_path.size(0),-1)
                merged_spatial_path = torch.cat((spatial_path, gobal_path), dim=1)
                merged_spatial_path = self.merge_layer(merged_spatial_path)


            else:
                global_path = self.global_pool(heatmap1)
                global_path = global_path.view(global_path.size(0), -1)  # Flatten
                global_path = torch.multiply(
                    global_path, torch.abs(global_path)
                )  # Average over the channels

                # Spatial Activity Path
                spatial_path = self.conv1(heatmap1)
                spatial_path = self.bn1(spatial_path)
                spatial_path = self.in1(
                    spatial_path
                )  # Uncomment if using instance normalization
                spatial_path = torch.nn.GELU(approximate="tanh")(spatial_path)
                # spatial_path = self.pool(spatial_path)

                spatial_path = self.conv2(spatial_path)
                spatial_path = self.bn2(spatial_path)
                spatial_path = self.in2(
                    spatial_path
                )  # Uncomment if using instance normalization
                spatial_path = torch.nn.GELU(approximate="tanh")(spatial_path)
                # spatial_path = self.pool(spatial_path)

                spatial_path = self.conv3(spatial_path)
                spatial_path = self.bn3(spatial_path)
                spatial_path = self.in3(
                    spatial_path
                )
                spatial_path = torch.nn.GELU(approximate="tanh")(spatial_path)

                spatial_path = self.conv4(spatial_path)
                spatial_path = self.bn4(spatial_path)
                spatial_path = self.in4(
                    spatial_path
                )
                spatial_path = torch.nn.GELU(approximate="tanh")(spatial_path)

                spatial_path = self.conv5(spatial_path)
                spatial_path = self.bn5(spatial_path)
                spatial_path = self.in5(
                    spatial_path
                )
                spatial_path = torch.nn.GELU(approximate="tanh")(spatial_path)

                spatial_path = spatial_path.view(spatial_path.size(0), -1)

                spatial_path = F.dropout(spatial_path, p=self.dropout_rate, training=self.training)  # Dropout
                spatial_path = self.fc1(spatial_path)
                spatial_path = F.dropout(spatial_path, p=self.dropout_rate, training=self.training)
                spatial_path = self.fc2(spatial_path)

                # heatmap1_only_channels = F.dropout(heatmap1_only_channels, p=self.dropout_rate, training=self.training)  # Dropout
                # channel_wise_spatial_path = self.fc_all1(heatmap1_only_channels)

                gobal_path = global_path.view(global_path.size(0), -1)
                # channel_wise_spatial_path = channel_wise_spatial_path.view(channel_wise_spatial_path.size(0),-1)
                merged_spatial_path = torch.cat((spatial_path, gobal_path), dim=1)
                merged_spatial_path = self.merge_layer(merged_spatial_path)

            return merged_spatial_path

    def train_model(self, train_loader, learning_rate=0.00001, epochs=10):
        if self.use_difference_heatmap:
            learning_rate = 0.0001
        self.train()

        optimizer = optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=0.1, amsgrad=True
        )
        # early_stopping = EarlyStopping(patience=5, min_delta=0.001)  # Adjust as needed

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * (10**1.5),
            total_steps=train_loader.__len__() * epochs,
            anneal_strategy="cos",
            three_phase=False,
            div_factor=10**1.5,
            final_div_factor=1e3,
        )
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)

        progress_bar = tqdm(train_loader, desc="Epochs", leave=False, total=epochs)
        total_loss = torch.tensor(0.0).to(self.device)
        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.float(), targets.float()  #
                if self.use_difference_heatmap:
                    targets, _ = torch.unbind(targets, dim=0)
                    heatmap1, heatmap2 = torch.unbind(inputs, dim=0)
                    if self.number_of_grids == 5:
                        heatmap1 = heatmap1.view(-1, 1, 16, 24)  # Reshape input
                        heatmap2 = heatmap2.view(-1, 1, 16, 24)  # Reshape inpu
                    else:
                        heatmap1 = heatmap1.view(
                            -1, 1, 8, 8 * self.number_of_grids
                        )  # Reshape input
                        heatmap2 = heatmap2.view(
                            -1, 1, 8, 8 * self.number_of_grids
                        )  # Reshape inpu

                    heatmap1 = heatmap1.to(self.device)
                    heatmap2 = heatmap2.to(self.device)
                else:
                    if not self.use_muovi_pro:
                        if self.number_of_grids == 5:
                            heatmap1 = inputs.view(-1, 1, 16, 24)  # Reshape input
                        else:
                            heatmap1 = inputs.view(-1, 1, 8, 8 * self.number_of_grids)
                    else:
                        heatmap1 = inputs.view(-1, 1, 2, 16)
                    heatmap1 = heatmap1.to(self.device)
                targets = targets.to(self.device)

                if self.use_difference_heatmap:
                    output = self(heatmap1, heatmap2)
                else:
                    output = self(heatmap1)

                for one_output in range(output.shape[1]):
                    loss_one_output = self.criterion(
                        output[:, one_output], targets[:, one_output]
                    )
                    total_loss += loss_one_output

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

                scheduler.step()

                total_loss = 0.0
            progress_bar.update(1)

        progress_bar.close()

    def predict(self, heatmap1, heatmap2=None):
        # TODO hier muss noch die grid aranger funktion rein
        with torch.no_grad():
            if not self.use_difference_heatmap:
                x = torch.from_numpy(heatmap1)
                x = x.float()  # Convert to float
                if not self.use_muovi_pro:
                    if self.number_of_grids == 5:
                        x = x.view(-1, 1, 16, 24)
                    else:
                        x = x.view(-1, 1, 8, 8 * self.number_of_grids)  # Reshape input
                else:
                    x = x.view(-1, 1, 2, 16)
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
                    x2 = x2.view(-1, 1, 16, 24)
                    x2 = x2.to(self.device)
                else:
                    x1 = x1.view(-1, 1, 8, 8 * self.number_of_grids)
                    x1 = x1.to(self.device)

                    x2 = torch.from_numpy(heatmap2)
                    x2 = x2.float()
                    x2 = x2.view(-1, 1, 8, 8 * self.number_of_grids)
                    x2 = x2.to(self.device)
                return self(x1, x2).cpu().numpy()

    def evaluate_best(self, predictions, ground_truth):
        self.eval()

        if isinstance(predictions, list):
            predictions = torch.tensor(np.array(predictions), dtype=torch.float32)
        if isinstance(ground_truth, list):
            ground_truth = torch.tensor(np.array(ground_truth), dtype=torch.float32)

        r_squared_values = []
        for i in range(
            predictions.shape[1]
        ):  # Assuming the second dimension contains the outputs
            sst = torch.sum((ground_truth[:, i] - torch.mean(ground_truth[:, i])) ** 2)
            ssr = torch.sum((ground_truth[:, i] - predictions[:, i]) ** 2)
            r_squared_values.append(1 - ssr / (sst + 1e-8))

        mse_loss = torch.tensor(0.0).to(self.device)
        for one_output in range(ground_truth.shape[1]):
            mse_loss_one_output = self.criterion(
                predictions[:, one_output], ground_truth[:, one_output]
            )
            mse_loss += mse_loss_one_output

        # Calculate the mean R-squared value
        mean_r_squared = torch.mean(torch.tensor(r_squared_values))
        print(f"Average Loss to next recording: {mse_loss}")
        return mean_r_squared.item(), mse_loss.item()

    def evaluate(self, test_loader):
        self.eval()
        total_loss = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.float(), targets.float()  #
                if self.use_difference_heatmap:
                    targets, _ = torch.unbind(targets, dim=0)
                    heatmap1, heatmap2 = torch.unbind(inputs, dim=0)
                    if self.number_of_grids == 5:
                        heatmap1 = heatmap1.view(-1, 1, 16, 24)
                        heatmap2 = heatmap2.view(-1, 1, 16, 24)
                    else:
                        heatmap1 = heatmap1.view(
                            -1, 1, 8, 8 * self.number_of_grids
                        )  # Reshape input
                        heatmap2 = heatmap2.view(
                            -1, 1, 8, 8 * self.number_of_grids
                        )  # Reshape inpu

                    heatmap1 = heatmap1.to(self.device)
                    heatmap2 = heatmap2.to(self.device)
                else:
                    if not self.use_muovi_pro:
                        if self.number_of_grids == 5:
                            heatmap1 = inputs.view(-1, 1, 16, 24)  # Reshape input
                        else:
                            heatmap1 = inputs.view(-1, 1, 8, 8 * self.number_of_grids)
                    else:
                        heatmap1 = inputs.view(-1, 1, 2, 16)
                    heatmap1 = heatmap1.to(self.device)

                targets = targets.to(self.device)
                if self.use_difference_heatmap:
                    outputs = self(heatmap1, heatmap2)
                else:
                    outputs = self(heatmap1)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        print(f"Average Loss Test Data: {avg_loss}")

        # Convert lists to numpy arrays for plotting
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)

        # # Plotting
        # plt.figure(figsize=(12, 6))
        #
        # for i in range(len(self.finger_indexes)):
        #     plt.subplot(1, len(self.finger_indexes), i + 1)
        #     plt.scatter(np.arange(len(all_targets)), all_targets[:, i], color='blue', label='True Values')
        #     plt.scatter(np.arange(len(all_predictions)), all_predictions[:, i], color='red', label='Predictions')
        #     for j in range(len(all_targets)):
        #         plt.plot([j, j], [all_targets[j, i], all_predictions[j, i]], color='gray', linestyle='--')
        #     plt.title(f'Comparison of True Values and Predictions (Output {i + 1})')
        #     plt.legend()
        #
        # plt.show()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load_model(cls, file_path):
        # model = ShallowConvNetWithAttention()
        cls.load_state_dict(torch.load(file_path))
        cls.eval()
        return cls

    def load_and_further_train(
        self, file_path, train_loader, additional_epochs=10, new_learning_rate=0.000001
    ):
        """
        Load a pre-trained model and further train it with given data.

        :param file_path: Path to the pre-trained model file.
        :param train_loader: DataLoader for the training data.
        :param additional_epochs: Number of additional epochs to train.
        :param new_learning_rate: Learning rate for further training.
        :return: Trained model.
        """
        self.train_model(
            train_loader, learning_rate=new_learning_rate, epochs=additional_epochs
        )
        return self

    def load_trainings_data(self, patient_number):
        adding = ""
        if self.retrain:
            adding = "_retrain" + str(self.retrain_number)
        if not self.use_mean:
            print("not using mean in Shallow Conv")
            X_test = np.array(
                helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/X_test_local"
                    + adding
                    + ".pkl"
                )
            )
            y_test = np.array(
                helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/y_test_local"
                    + adding
                    + ".pkl"
                )
            )
            X_train = np.array(
                helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/X_train_local"
                    + adding
                    + ".pkl"
                )
            )
            y_train = np.array(
                helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/y_train_local"
                    + adding
                    + ".pkl"
                )
            )
        else:
            print("Using mean in Shallow Conv")
            X_test = np.array(
                helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/X_test_local_mean"
                    + adding
                    + ".pkl"
                )
            )
            y_test = np.array(
                helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/y_test_local_mean"
                    + adding
                    + ".pkl"
                )
            )
            X_train = np.array(
                helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/X_train_local_mean"
                    + adding
                    + ".pkl"
                )
            )
            y_train = np.array(
                helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/y_train_local_mean"
                    + adding
                    + ".pkl"
                )
            )

        X_train = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(
            X_train
        ).transpose(
            2, 0, 1
        )
        X_test = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(
            X_test
        ).transpose(
            2, 0, 1
        )
        # y_train = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(y_train).transpose(2,1,0)
        # y_test = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(y_test).transpose(2,1,0)

        if not self.use_difference_heatmap:
            # Create the data loaders
            train_dataset = TensorDataset(
                torch.from_numpy(X_train), torch.from_numpy(y_train)
            )
            test_dataset = TensorDataset(
                torch.from_numpy(X_test), torch.from_numpy(y_test)
            )
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=True
            )
            self.train_loader = train_loader
            self.test_loader = test_loader
            return train_loader, test_loader

        else:  # self.use_difference_heatmap:
            if not self.use_mean:
                self.training_data_time = helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/training_data_time"
                    + adding
                    + ".pkl"
                )[self.best_time_index]
            else:
                self.training_data_time = helpers.load_pickle_file(
                    r"trainings_data/resulting_trainings_data/subject_"
                    + str(patient_number)
                    + "/training_data_time_mean"
                    + adding
                    + ".pkl"
                )[self.best_time_index]
            # X_train, X_test, y_train, y_test is order in time data
            X_train_time = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(
                self.training_data_time[0]
            ).transpose(
                2, 0, 1
            )
            X_test_time = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(
                self.training_data_time[1]
            ).transpose(
                2, 0, 1
            )
            # y_train_time = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(self.training_data_time[2]).transpose(2,1,0)
            # y_test_time = self.grid_aranger.transfer_and_concatenate_320_into_grid_arangement_all_samples(self.training_data_time[3]).transpose(2,1,0)
            y_train_time = self.training_data_time[2]
            y_test_time = self.training_data_time[3]

            train_dataset_time = TensorDataset(
                torch.from_numpy(np.stack((X_train, X_train_time), axis=0)),
                torch.from_numpy(np.stack((y_train, y_train_time), axis=0)),
            )
            test_dataset_time = TensorDataset(
                torch.from_numpy(np.stack((X_test, X_test_time), axis=0)),
                torch.from_numpy(np.stack((y_test, y_test_time), axis=0)),
            )

            train_loader_time = DataLoader(
                train_dataset_time, batch_size=self.batch_size, shuffle=True
            )
            test_loader_time = DataLoader(
                test_dataset_time, batch_size=self.batch_size, shuffle=True
            )
            self.train_loader = train_loader_time
            self.test_loader = test_loader_time
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

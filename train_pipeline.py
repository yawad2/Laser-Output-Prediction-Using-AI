"""
A pipeline for training U-net using different loss functions.
"""
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import os
import matplotlib.pyplot as plt
from misc_utils import *
from unet_arch import Generator

data_path = './spatial'

# Get train/test tensors (split by run)
X_train_np, y_train_np, X_test_np, y_test_np, train_subruns, test_subruns, test_files_paths = get_torch_dataset_split_by_run(data_path)
# X_train_np, y_train_np = duplicate_low_subrun_samples(X_train_np, y_train_np, train_subruns)

X_train = torch.from_numpy(X_train_np).float()
y_train = torch.from_numpy(y_train_np).float()
X_test = torch.from_numpy(X_test_np).float()    
y_test = torch.from_numpy(y_test_np).float()

# Create Datasets and Dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define loss functions to be used
train_losses = {
    "Hybrid (0.96 L1, 0.04 abs. UV error)": hybrid_loss
}

# Set device and check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for loss_name, loss_fn in train_losses.items():

    directory = f"{loss_name}_model_by_run_split"
    os.makedirs(directory, exist_ok=True)
    
    # Initialize model, optimizer, and other training components
    model = Generator()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    num_epochs = 300
    model.to(device)
    
    # Store losses 
    train_mse_per_epoch = [] 
    test_mse_per_epoch = [] 

    train_L1_per_epoch = []  
    test_L1_per_epoch = [] 

    train_huber_per_epoch = []  
    test_huber_per_epoch = []  

    train_mape_per_epoch = [] 
    test_mape_per_epoch = [] 

    avg_losses_per_epoch = [] 
    
    # Training loop
    print(f"Training with {loss_name} loss...")
    start_time = datetime.now()
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_epoch_loss = np.mean(epoch_losses)
        avg_losses_per_epoch.append(avg_epoch_loss)
        train_mse, train_l1, train_huber, train_mape = evaluate(model, train_loader, device)
        test_mse, test_l1, test_huber, test_mape = evaluate(model, test_loader, device)
        train_mse_per_epoch.append(train_mse)
        test_mse_per_epoch.append(test_mse)
        train_L1_per_epoch.append(train_l1)
        test_L1_per_epoch.append(test_l1)
        train_huber_per_epoch.append(train_huber)
        test_huber_per_epoch.append(test_huber)
        train_mape_per_epoch.append(train_mape)
        test_mape_per_epoch.append(test_mape)

    end_time = datetime.now()
    print(f"Training completed in: {end_time - start_time}")
    
    # Evaluation on test set
    mse, l1, huber_loss, mape = evaluate(model, test_loader, device)
    print(f"Test MSE: {mse:.4f}, L1: {l1:.4f}, Huber: {huber_loss:.4f}, MAPE: {mape:.4f}")
    
    # Evaluation on train set
    train_mse, train_l1, train_huber, train_mape = evaluate(model, train_loader, device)
    print(f"Final train MSE: {train_mse:.4f}, L1: {train_l1:.4f}, Huber Loss: {train_huber:.4f}, MAPE: {train_mape:.4f}")
    
    # # Save training metrics
    # metrics = {
    #     "train_mse_per_epoch": train_mse_per_epoch,
    #     "test_mse_per_epoch": test_mse_per_epoch,
    #     "train_L1_per_epoch": train_L1_per_epoch,
    #     "test_L1_per_epoch": test_L1_per_epoch,
    #     "train_huber_per_epoch": train_huber_per_epoch,
    #     "test_huber_per_epoch": test_huber_per_epoch,
    #     "train_mape_per_epoch": train_mape_per_epoch,
    #     "test_mape_per_epoch": test_mape_per_epoch
    # }
    # metrics_df = pd.DataFrame(metrics)
    # metrics_df.to_csv(f"./{directory}/{loss_name}_metrics_by_run_split.csv", index=False)

    # Generate and save plots
    get_plot_two_lines(
        data1=train_mse_per_epoch, 
        data2=test_mse_per_epoch, 
        label1="Train MSE", 
        label2="Test MSE", 
        x_label="Epochs", 
        y_label="MSE Loss", 
        title=f"{loss_name} Loss - MSE per Epoch",
        save_path=f"./{directory}/{loss_name}_mse_vs_epoch.png"
    )

    get_plot_two_lines(
        data1=train_L1_per_epoch, 
        data2=test_L1_per_epoch, 
        label1="Train L1", 
        label2="Test L1", 
        x_label="Epochs", 
        y_label="L1 Loss", 
        title=f"{loss_name} Loss - L1 per Epoch",
        save_path=f"./{directory}/{loss_name}_l1_vs_epoch.png"
    )

    get_plot_two_lines(
        data1=train_huber_per_epoch, 
        data2=test_huber_per_epoch, 
        label1="Train Huber", 
        label2="Test Huber", 
        x_label="Epochs", 
        y_label="Huber Loss", 
        title=f"{loss_name} Loss - Huber per Epoch",
        save_path=f"./{directory}/{loss_name}_huber_vs_epoch.png"
    )
    get_plot_two_lines(
        data1=train_mape_per_epoch, 
        data2=test_mape_per_epoch, 
        label1="Train MAPE", 
        label2="Test MAPE", 
        x_label="Epochs", 
        y_label="MAPE Loss", 
        title=f"{loss_name} Loss - MAPE per Epoch",
        save_path=f"./{directory}/{loss_name}_mape_vs_epoch.png"
    )

    get_plot_one_line(
        data=avg_losses_per_epoch, 
        label="Training Loss", 
        x_label="Epochs", 
        y_label="Loss", 
        title=f"{loss_name} Loss - Training Loss per Epoch",
        save_path=f"./{directory}/{loss_name}_training_loss.png"
    )
 
    # Save the model
    torch.save(model.state_dict(), f"./{directory}/{loss_name}_model_by_run_split.pth")
    print(f"Model saved to {directory}/{loss_name}_model_by_run_split.pth")

    # Save predictions & compute UV % error on test set
    print("Saving test inputs, predictions, and targets...")

    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        inputs_np = X_test.cpu().numpy()
        targets_np = y_test.cpu().numpy()
        energy_diff_percent = compute_energy_difference_percentage(preds[:, 0], targets_np[:, 0])

    avg_energy_diff_low = np.mean(energy_diff_percent[test_subruns <= 7])
    avg_energy_diff_high = np.mean(energy_diff_percent[test_subruns > 7])

    # Extract run/subrun for each test sample
    run_subrun_list = [extract_run_subrun(path) for path in test_files_paths]
    test_subruns = np.array(test_subruns)

    # plot energy difference % vs subrun
    sorted_indices = np.argsort(test_subruns)
    sorted_subruns = test_subruns[sorted_indices]
    sorted_energy_diff = energy_diff_percent[sorted_indices]

    plt.figure()
    plt.scatter(sorted_subruns, sorted_energy_diff, alpha=0.7, c='blue', label='UV Energy Diff %')
    # plt.axvline(x=7, color='red', linestyle='--', label='Subrun = 7 Threshold')
    plt.title("UV Energy Difference % vs Subrun (Test Set)")
    plt.xlabel("Subrun")
    plt.ylabel("UV Energy Difference %")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./{directory}/energy_diff_vs_subrun.png")
    print(f"Saved Energy Diff vs Subrun plot to ./{directory}/energy_diff_vs_subrun.png")
    save_inp_pred_tar(inputs_np[:, 0], preds[:, 0], targets_np[:, 0], run_subrun_list, destination_folder=f"./{directory}/test_outputs")
    print(f"Training with {loss_name} loss completed and results saved in {directory}.\n")
    print("-" * 110)

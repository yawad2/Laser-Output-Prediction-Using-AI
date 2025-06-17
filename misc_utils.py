import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import os
import pandas as pd
import torch.nn as nn
import re
from sklearn.model_selection import train_test_split

# def custom_loss_fn(pred, target):
# 	joules_Scaler = 0.0333958286584664
# 	pred_energy = torch.sum(pred) * joules_Scaler
# 	target_energy = torch.sum(target) * joules_Scaler
# 	return torch.abs(target_energy - pred_energy)

epsilon = 1e-8 
mse = nn.MSELoss()

# def dice_loss(pred, target, smooth=1.0):
#     intersection = (pred * target).sum(dim=(2, 3))
#     union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
#     # compute dice coefficient
#     dice = (2. * intersection + smooth) / (union + smooth)
#     return 1 - dice.mean()  # return the loss


def log_cosh_loss(pred, target):
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff) + epsilon))

def RMSLE_loss(pred, target):
    pred_clamped = torch.clamp(pred, min=0)
    target_clamped = torch.clamp(target, min=0)
    return torch.sqrt(mse(torch.log(pred_clamped + 1), torch.log(target_clamped + 1)) + epsilon)

def focal_huber_loss(pred, target, degree=4, delta=1.0):
    error = pred - target
    abs_error = torch.abs(error)
    degree_term = torch.min(abs_error, torch.tensor(delta, device=abs_error.device, dtype=abs_error.dtype))
    linear = abs_error - degree_term
    loss = 0.5 * degree_term ** degree + delta * linear
    return torch.mean(loss)

def energy_per_loss(pred, target):
    return (get_energy_tensor(target) - get_energy_tensor(pred)) / (get_energy_tensor(target)+epsilon)

def abs_energy_per_loss(pred, target):
    return torch.abs(get_energy_tensor(target) - get_energy_tensor(pred)) / (get_energy_tensor(target)+ epsilon)

def MARE_loss(output, target):
    return torch.mean(torch.abs(target - output) / (torch.abs(target) + epsilon))

def MAPE_loss(output, target):
    return torch.mean(torch.abs((target - output) / (target+epsilon))) * 100

def get_heatmaps(images, titles, colorscale=True, cbar_title='Intensity'):
    fig = plt.figure(figsize=(16, 4))  # Adjust the figure size as needed
    gs = gridspec.GridSpec(1, len(images) + 1, width_ratios=[1]*len(images) + [0.05])

    # Global min and max for consistent color scaling (using the ground truth image as reference)
    global_min = np.min(images[1])
    global_max = np.max(images[1])

    for i, img in enumerate(images):
        ax = fig.add_subplot(gs[i])
        if i == 0:  # No scaling for the input image
            im = ax.imshow(img, cmap='jet')
        else:  # Scaled images
            im = ax.imshow(img, cmap='jet', vmin=global_min, vmax=global_max)
        ax.set_title(titles[i])
        ax.axis('off')

    if colorscale:
        cbar_ax = fig.add_subplot(gs[-1])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(cbar_title)

    plt.show()


def get_energy(image):
    joules_Scaler = 0.0333958286584664
    energy = np.sum(image) * joules_Scaler
    return energy

def get_energy_tensor(image):
    joules_Scaler = 0.0333958286584664
    energy = torch.sum(image) * joules_Scaler
    return energy

def save_inp_pred_tar(inputs, preds, targets, run_subrun_list, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

    for i in range(len(inputs)):
        input_img = inputs[i]
        pred_img = preds[i]
        target_img = targets[i]
        run, subrun = run_subrun_list[i]

        input_filename = f"{destination_folder}/input_sample_run_{run}_subrun_{subrun}.csv"
        pred_filename = f"{destination_folder}/pred_sample_run_{run}_subrun_{subrun}.csv"
        target_filename = f"{destination_folder}/target_sample_run_{run}_subrun_{subrun}.csv"

        pd.DataFrame(pred_img).to_csv(pred_filename, index=False, header=False)
        pd.DataFrame(target_img).to_csv(target_filename, index=False, header=False)
        pd.DataFrame(input_img).to_csv(input_filename, index=False, header=False)

    print(f"Saved inputs, predictions, and targets in {destination_folder}")

def predict(model, inputs, device):
    model.eval()
    inputs_tensor = torch.from_numpy(inputs).float().to(device)
    with torch.no_grad():
        outputs = model(inputs_tensor)
    outputs = outputs.cpu().numpy()
    return outputs

def get_df_3images(folder_path):
    predictions_files_list = [f for f in os.listdir(folder_path) if "pred_" in f]
    targets_files_list = [f for f in os.listdir(folder_path) if "target_" in f]
    input_files_list = [f for f in os.listdir(folder_path) if "input_" in f]
    
    predictions = [pd.read_csv(os.path.join(folder_path, filename), header=None) for filename in predictions_files_list]
    ground_truth = [pd.read_csv(os.path.join(folder_path, filename), header=None) for filename in targets_files_list]
    inputs = [pd.read_csv(os.path.join(folder_path, filename), header=None) for filename in input_files_list]
    
    # Combine into a single DataFrame
    df = pd.DataFrame({
        'prediction': predictions,
        'target': ground_truth,
        'input': inputs
    })
    return df

def extract_run_subrun(path):
    match = re.search(r'Run_(\d+)_.*InjEnergyFactor_(\d+)', path)
    if match:
        run = int(match.group(1))
        subrun = int(match.group(2))
        return run, subrun
    else:
        return None

def load_data_from_folders(folder_list):
    input_data = []
    output_data = []
    subruns = []
    test_files_paths = []
    for input_paths, output_paths in folder_list:
        for in_file, out_file in zip(input_paths, output_paths):
            input_array = pd.read_csv(in_file, header=None).values
            output_array = pd.read_csv(out_file, header=None).values
            input_data.append(input_array)
            output_data.append(output_array)
            _, subrun = extract_run_subrun(in_file)
            subruns.append(subrun)
            test_files_paths.append(in_file)

    input_data = np.expand_dims(np.array(input_data), axis=1)
    output_data = np.expand_dims(np.array(output_data), axis=1)
    subruns_array = np.array(subruns)
    return input_data, output_data, subruns_array, test_files_paths

def duplicate_low_subrun_samples(X_train_np, y_train_np, subruns_train, threshold=7, num_duplicates=3):
    low_indices = np.where(subruns_train <= threshold)[0]
    print(f"Number of train samples with subrun <= {threshold}: {len(low_indices)}")
    print(f"Number of train samples with subrun > {threshold}: {subruns_train.shape[0] - len(low_indices)}")
    
    # Get samples 
    X_low = X_train_np[low_indices]
    y_low = y_train_np[low_indices]

    # Duplicate samples
    X_dup = np.concatenate([X_low] * num_duplicates, axis=0)
    y_dup = np.concatenate([y_low] * num_duplicates, axis=0)

    # Concatenate duplicated samples to the original
    X_aug = np.concatenate([X_train_np, X_dup], axis=0)
    y_aug = np.concatenate([y_train_np, y_dup], axis=0)

    print("Train shape after duplication:", X_aug.shape, y_aug.shape)
    return X_aug, y_aug


def get_torch_dataset_split_by_run(master_folder):
    # Load and split data by folder
    subfolder_data = []

    for subfolder in os.listdir(master_folder):
        subfolder_path = os.path.join(master_folder, subfolder)
        if os.path.isdir(subfolder_path):
            input_files = sorted([f for f in os.listdir(subfolder_path) if "Inj_256x256_InjEnergyFactor_" in f and f.endswith(".csv")])
            output_files = sorted([f for f in os.listdir(subfolder_path) if "UV_256x256_InjEnergyFactor_" in f and f.endswith(".csv")])
            input_paths = [os.path.join(subfolder_path, f) for f in input_files]
            output_paths = [os.path.join(subfolder_path, f) for f in output_files]
            if input_paths and output_paths:
                subfolder_data.append((input_paths, output_paths))

    print(f"Number of subfolders: {len(subfolder_data)}")

    # Split subfolders
    train_folders, test_folders = train_test_split(subfolder_data, test_size=0.2, random_state=42)

    # Load and convert data to tensors
    X_train_np, y_train_np, train_subruns, _  = load_data_from_folders(train_folders)
    X_test_np, y_test_np, test_subruns, test_files_paths = load_data_from_folders(test_folders)

    print("Train shape:", X_train_np.shape, y_train_np.shape)
    print("Test shape:", X_test_np.shape, y_test_np.shape)

    return X_train_np, y_train_np, X_test_np, y_test_np, train_subruns, test_subruns, test_files_paths


def plot_region_losses(train_losses, test_losses, train_subruns, test_subruns, loss_name, save_path):
    # Convert to numpy arrays for indexing
    train_subruns = np.array(train_subruns)
    test_subruns = np.array(test_subruns)

    # Prepare full train and test losses for each subrun
    train_full = np.concatenate([train_losses[f"{loss_name}_low"], train_losses[f"{loss_name}_high"]])
    test_full = np.concatenate([test_losses[f"{loss_name}_low"], test_losses[f"{loss_name}_high"]])
    train_subrun_full = np.concatenate([train_subruns[train_subruns <= 7], train_subruns[train_subruns > 7]])
    test_subrun_full = np.concatenate([test_subruns[test_subruns <= 7], test_subruns[test_subruns > 7]])

    # Compute average loss per subrun
    def average_by_subrun(losses, subruns):
        unique_subruns = np.unique(subruns)
        avg_losses = []
        for subrun in unique_subruns:
            avg_losses.append(np.mean(losses[subruns == subrun]))
        return unique_subruns, np.array(avg_losses)

    train_subrun_vals, train_avg = average_by_subrun(train_full, train_subrun_full)
    test_subrun_vals, test_avg = average_by_subrun(test_full, test_subrun_full)

    # Sort by subrun for plotting
    train_sort_idx = np.argsort(train_subrun_vals)
    test_sort_idx = np.argsort(test_subrun_vals)

    plt.figure()
    plt.plot(train_subrun_vals[train_sort_idx], train_avg[train_sort_idx], label="Train Loss")
    plt.plot(test_subrun_vals[test_sort_idx], test_avg[test_sort_idx], label="Test Loss")
    plt.xlabel("Subrun")
    plt.ylabel(f"{loss_name} Loss")
    plt.title(f"{loss_name} Loss by Subrun (Train vs Test)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined plot to {save_path}")

def evaluate(model, dataloader, device):
    model.eval()
    mse_loss_fn = nn.MSELoss()
    l1_loss_fn = nn.L1Loss()
    huber_loss_fn = nn.HuberLoss(delta=1.0)
    mape_loss_fn = MAPE_loss


    total_mse = 0.0
    total_l1 = 0.0
    total_huber = 0.0
    total_mape = 0.0
    total_samples = 0

    with torch.no_grad():
        for input_image, target in dataloader:
            input_image = input_image.to(device)
            target = target.to(device)

            gen_output = model(input_image)
            batch_size = input_image.size(0)

            total_mse += mse_loss_fn(gen_output, target).item() * batch_size
            total_l1 += l1_loss_fn(gen_output, target).item() * batch_size
            total_huber += huber_loss_fn(gen_output, target).item() * batch_size
            total_mape += mape_loss_fn(gen_output, target).item() * batch_size

            total_samples += batch_size

    # Average loss over all samples
    mse = total_mse / total_samples
    l1 = total_l1 / total_samples
    huber = total_huber / total_samples
    mape = total_mape / total_samples

    return mse, l1, huber, mape


def get_plot_two_lines(data1, label1, data2, label2, x_label, y_label, title, save_path):
    plt.figure()
    plt.plot(data1, label = label1)
    plt.plot(data2, label = label2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f'Avg. {label1}: {np.mean(data1)}')
    print(f'Avg. {label2}: {np.mean(data2)}')


def get_plot_one_line(data, label, x_label, y_label, title, save_path):
    plt.figure()
    plt.plot(data, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f'Avg. {label}: {np.mean(data)}')


def evaluate_by_region(model, inputs, targets, subruns, device):
    model.eval()
    inputs_tensor = torch.from_numpy(inputs).float().to(device)
    targets_tensor = torch.from_numpy(targets).float().to(device)

    with torch.no_grad():
        preds = model(inputs_tensor).cpu().numpy()
        targets_np = targets_tensor.cpu().numpy()

    subruns = np.array(subruns)
    losses = {"mse": [], "l1": [], "huber": [], "mape": []}
    low_region = (subruns <= 7)
    high_region = (subruns > 7)

    def calc_losses(pred, target):
        return {
            "mse": np.mean((pred - target) ** 2),
            "l1": np.mean(np.abs(pred - target)),
            "huber": np.mean(np.where(np.abs(pred - target) < 1, 0.5 * (pred - target) ** 2, np.abs(pred - target) - 0.5)),
            "mape": np.mean(np.abs((pred - target) / (target + 1e-8)))
        }

    for region, name in [(low_region, "low"), (high_region, "high")]:
        region_losses = {"mse": [], "l1": [], "huber": [], "mape": []}
        for i in np.where(region)[0]:
            result = calc_losses(preds[i, 0], targets_np[i, 0])
            for k in region_losses:
                region_losses[k].append(result[k])
        for k in ['mse', 'l1', 'huber', 'mape']:
            losses[k + f"_{name}"] = np.array(region_losses[k])
    return losses

def compute_energy_difference_percentage(preds, targets):
    joules_scaler = 0.0333958286584664
    pred_energy = np.sum(preds, axis=(1, 2)) * joules_scaler
    target_energy = np.sum(targets, axis=(1, 2)) * joules_scaler
    diff_percent = ((target_energy - pred_energy) / (target_energy + 1e-8)) * 100
    return diff_percent


def save_heatmaps(images, titles, path, colorscale=True, cbar_title='Intensity'):
    fig = plt.figure(figsize=(16, 4))  # Adjust the figure size as needed
    gs = gridspec.GridSpec(1, len(images) + 1, width_ratios=[1]*len(images) + [0.05])

    # Global min and max for consistent color scaling (using the ground truth image as reference)
    global_min = np.min(images[0])
    global_max = np.max(images[0])

    for i, img in enumerate(images):
        ax = fig.add_subplot(gs[i])
        if i == 0:  # No scaling for the input image
            im = ax.imshow(img, cmap='jet')
        else:  # Scaled images
            im = ax.imshow(img, cmap='jet', vmin=global_min, vmax=global_max)
        ax.set_title(titles[i])
        ax.axis('off')

    # if colorscale:
        # cbar_ax = fig.add_subplot(gs[-1])
        # cbar = fig.colorbar(im, cax=cbar_ax)
        # cbar.set_label(cbar_title)
    
    run, subrun = extract_run_subrun_df(path)
    fig.suptitle(f"Run {run}, Subrun {subrun}", fontsize=16)

    fig.subplots_adjust(wspace=0.1, top=0.85)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

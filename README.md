## Overview

This repository implements a U-Net architecture and a training pipeline tailored for energy-based laser prediction.

### `utils/unet_arch.py`
Defines the U-Net architecture as presented in the [published work](https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-24-42692&id=562704).

### `utils/misc_utils.py`
Contains utility functions for:
- Data loading  
- Energy calculation  
- Custom loss functions  
- Visualization and plotting  

### `train_pipeline.py`
Main training script that:
- Iterates over a dictionary of loss functions  
- Saves trained models and loss plots  
- Exports input, target, and prediction data as CSV files  
- Organizes outputs in a structured directory per experiment

### `eval/visualize_images.ipynb`
Contains logic for:
- Saving heatmaps for input, target, prediction, and difference (target - prediction) from csv files
- Generating video for a folder of images (eg. a video of all subruns' heatmaps for a given run)
- Generating boxplots for the energy % error


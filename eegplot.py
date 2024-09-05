# plot_eeg_signals.py

import matplotlib.pyplot as plt
import scipy.io
import numpy as np

def load_data(file_path):
    """
    Load EEG data from a .mat file.

    Parameters:
    file_path (str): Path to the .mat file containing the EEG data.

    Returns:
    eeg_data (numpy array): EEG data with shape (num_samples, num_channels).
    """
    data = scipy.io.loadmat(file_path)
    eeg_data = data['o']['data'][0][0]  # Adjust for the structure of your data
    return eeg_data

def plot_eeg_signals(file_path):
    """
    Plot EEG signals from a .mat file.

    Parameters:
    file_path (str): Path to the .mat file containing the EEG data.
    """
    eeg_data = load_data(file_path)
    num_channels = eeg_data.shape[1]
    sampling_rate = 128  # Assuming a sampling rate of 128 Hz

    fig, axs = plt.subplots(num_channels, 1, figsize=(12, 6 * num_channels))
    for i in range(num_channels):
        channel_data = eeg_data[:, i]
        time = np.arange(len(channel_data)) / sampling_rate
        axs[i].plot(time, channel_data)
        axs[i].set_title(f"Channel {i+1}")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Amplitude (μV)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "EEG Data\eeg_record1.mat"  # Replace with your .mat file path
    plot_eeg_signals(file_path)
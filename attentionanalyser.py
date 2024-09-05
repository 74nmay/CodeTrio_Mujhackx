import numpy as np
import scipy
from scipy.signal import butter, lfilter, stft
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os  

# Load the Matlab files and extract the EEG data
def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    eeg_data = data['o']['data'][0][0]  # Adjust for the structure of your data
    return eeg_data

# Preprocess the data
def preprocess_data(eeg_data):
    eeg_data = np.nan_to_num(eeg_data)
    nyq = 0.5 * 128  # Nyquist frequency
    low = 0.5 / nyq  # Low cutoff frequency
    high = 30 / nyq  # High cutoff frequency
    b, a = butter(5, [low, high], btype='band')
    eeg_data = lfilter(b, a, eeg_data, axis=0)
    return eeg_data

# Extract time-frequency features using STFT
def extract_time_frequency_features(eeg_data, fs, nperseg, noverlap):
    features = []
    for i in range(eeg_data.shape[1]):
        channel_data = eeg_data[:, i]
        nperseg = min(nperseg, len(channel_data))
        noverlap = min(noverlap, nperseg - 1)
        f, t, Zxx = stft(channel_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
        features.append(np.mean(np.abs(Zxx)))
    return features
# Extract frequency band features and classify mental state
def extract_frequency_band_features(eeg_data):
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30)
    }
    state = ""
    features = []
    
    for i in range(eeg_data.shape[1]):
        channel_data = eeg_data[:, i]
        fft_data = np.fft.fft(channel_data)
        freqs = np.fft.fftfreq(len(channel_data), 1 / 128)  # Frequency bins based on 128 Hz sampling rate
        
        band_powers = {}
        for band, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_data = fft_data[band_mask]
            band_powers[band] = np.mean(np.abs(band_data))
        
        # Calculate alpha-beta ratio and theta-beta ratio
        alpha_beta_ratio = band_powers['Alpha'] / band_powers['Beta']
        theta_beta_ratio = band_powers['Theta'] / band_powers['Beta']
        
        # Classify mental state based on the ratios
        if alpha_beta_ratio > 1 and theta_beta_ratio > 1:
            state = "Unfocused"
        else:
            state = "Focused"
        
        print(f"Channel {i + 1}: Alpha-Beta Ratio: {alpha_beta_ratio:.2f}, Theta-Beta Ratio: {theta_beta_ratio:.2f}, State: {state}")
        
        features.append(list(band_powers.values()))  # Store the power of all bands
    
    return features, state

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Main function
def main():
    data_dir = 'EEG Data'
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]

    # Initialize lists to store the features
    time_frequency_features_list = []
    frequency_band_features_list = []
    labels = []

    # Loop through each EEG data file
    for file in data_files:
        file_path = os.path.join(data_dir, file)
        eeg_data = load_data(file_path)

        print("eeg_data shape:", eeg_data.shape)

        # Preprocess the data
        eeg_data = preprocess_data(eeg_data)

        # Parameters
        fs = 128
        nperseg = 25
        noverlap = nperseg // 2

        # Extract features
        time_frequency_features = extract_time_frequency_features(eeg_data, fs, nperseg, noverlap)
        frequency_band_features, state = extract_frequency_band_features(eeg_data)

        # Append features and labels
        time_frequency_features_list.append(time_frequency_features)
        frequency_band_features_list.append(frequency_band_features)
        
        # Append corresponding label for the state
        if state == "Focused":
            labels.append(0)
        else:  # Unfocused
            labels.append(1)

    # Combine features
    time_frequency_features_array = np.array(time_frequency_features_list)
    frequency_band_features_array = np.array(frequency_band_features_list)
    features = np.concatenate((time_frequency_features_array, frequency_band_features_array), axis=1)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    return model

if __name__ == "__main__":
    model = main()
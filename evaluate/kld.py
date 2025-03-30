import numpy as np
import librosa
import glob
import scipy

def extract_mfcc(audio_path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # Mean across time

def compute_histograms(mfcc_features, bins=50):
    return [np.histogram(mfcc, bins=bins, density=True)[0] for mfcc in mfcc_features]

def kl_divergence(p, q, epsilon=1e-10):
    """
    Compute KL divergence with smoothing to avoid log(0).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p += epsilon
    q += epsilon

    p /= np.sum(p)
    q /= np.sum(q)

    return scipy.stats.entropy(p, q)  # KL(P || Q)

set_A = "./samples/audios"
set_B = "./wav/ours"
set_A_files = glob.glob(f"{set_A}/*.wav")
set_B_files = glob.glob(f"{set_B}/*.wav")

set_A_mfccs = [extract_mfcc(f) for f in set_A_files]
set_B_mfccs = [extract_mfcc(f) for f in set_B_files]

set_A_histograms = compute_histograms(set_A_mfccs)
set_B_histograms = compute_histograms(set_B_mfccs)

kl_matrix = np.zeros((len(set_A_histograms), len(set_B_histograms)))
for i, hist_A in enumerate(set_A_histograms):
    for j, hist_B in enumerate(set_B_histograms):
        kl_matrix[i, j] = kl_divergence(hist_A, hist_B)

mean_kl = np.mean(kl_matrix)
print(f"Mean KL Divergence: {mean_kl}")
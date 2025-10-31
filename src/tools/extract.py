import os.path
import datetime as dt

import h5py
import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

import data_augmentation as aug
from src.tools import utils



def extract_dataset(dataset_path,
                    file_names,
                    extractor,
                    clip_duration,
                    output_path,
                    recompute=False,
                    n_transforms_iter=None,
                    device='cpu'):
    """Extract features from the audio clips in a dataset.

    Args:
        dataset_path (str): Path of directory containing dataset.
        file_names (list): List of file names for the audio clips.
        extractor: Class instance for feature extraction.
        clip_duration: Duration of a reference clip in seconds. Used to
            ensure all feature vectors are of the same length.
        output_path: File path of output HDF5 file.
        recompute (bool): Whether to extract features that already exist
            in the HDF5 file.
        n_transforms_iter (iterator): Iterator for the number of
            transformations to apply for each example. If data
            augmentation should be disabled, set this to ``None``.
            Otherwise, ensure that `file_names` has been expanded as if
            by calling :func:`data_augmentation.expand_metadata`.
        device (str): Device to use for computation ('cpu' or 'cuda').
    """
    extractor.to(device)

    # Create/load the HDF5 file to store the feature vectors
    with h5py.File(output_path, 'a') as f:
        size = len(file_names)  # Size of dataset

        # Create/load feature vector dataset and timestamp dataset
        feats_shape = (size,) + extractor.output_shape(clip_duration)
        feats = f.require_dataset('F', feats_shape, dtype=np.float32)
        timestamps = f.require_dataset('timestamps', (size,),
                                       dtype=h5py.special_dtype(vlen=bytes))

        transforms = iter(())

        for i, name in enumerate(tqdm(file_names)):
            # Skip if existing feature vector should not be recomputed
            if timestamps[i] and not recompute:
                next(transforms, None)
                continue

            # Generate next transform or, if iterator is empty, load
            # the next audio clip from disk. Note that the iterator will
            # always be empty if data augmentation (DA) is disabled.
            x = next(transforms, None)
            if x is None:
                # Load audio file from disk
                path = os.path.join(dataset_path, name)
                x, sample_rate = librosa.load(path, sr=None)
                x = torch.from_numpy(x).float().to(device)

                # Create new transform generator if DA is enabled
                if n_transforms_iter:
                    transforms = aug.transformations(
                        x, sample_rate, next(n_transforms_iter))

            # Compute feature vector using extractor
            vec = extractor.extract(x, sample_rate)
            vec = utils.pad_truncate(vec, feats_shape[1])

            # Convert to numpy for saving
            if isinstance(vec, torch.Tensor):
                vec = vec.cpu().numpy()

            # Save to dataset
            feats[i] = vec
            # Record timestamp in ISO format
            timestamps[i] = dt.datetime.now().isoformat()


def load_features(path):
    """Load feature vectors from the specified HDF5 file.

    Args:
        path (str): Path to the HDF5 file.

    Returns:
        torch.Tensor: Tensor of feature vectors.
    """
    with h5py.File(path, 'r') as f:
        features = np.array(f['F'])
        return torch.from_numpy(features).float()


class LogmelExtractor(nn.Module):
    """Feature extractor for logmel representations.

    A logmel feature vector is a spectrogram representation that has
    been scaled using a Mel filterbank and a log nonlinearity.

    Args:
        sample_rate (number): Target resampling rate.
        n_window (int): Number of bins in each spectrogram frame.
        hop_length (int): Number of samples between frames.
        n_mels (int): Number of Mel bands.

    Attributes:
        sample_rate (number): Target resampling rate.
        n_window (int): Number of bins in each spectrogram frame.
        hop_length (int): Number of samples between frames.
        n_mels (int): Number of Mel bands.
        mel_scale (torchaudio.transforms.MelScale): Mel scale transformer.
    """

    def __init__(self,
                 sample_rate=16000,
                 n_window=1024,
                 hop_length=512,
                 n_mels=64,
                 ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_window = n_window
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Create Mel filterbank using torchaudio
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_stft=n_window // 2 + 1
        )

        # Create resampler for potential sample rate conversion
        self.resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sample_rate)

    def output_shape(self, clip_duration):
        """Determine the shape of a logmel feature vector.

        Args:
            clip_duration (number): Duration of the input time-series
                signal given in seconds.

        Returns:
            tuple: The shape of a logmel feature vector.
        """
        n_samples = int(clip_duration * self.sample_rate)
        n_frames = (n_samples - self.n_window) // self.hop_length + 1
        return (n_frames, self.n_mels)

    def forward(self, x, sample_rate=None):
        """Transform the given signal into a logmel feature vector.

        Args:
            x (torch.Tensor): Input time-series signal of shape (batch_size, n_samples)
                            or (n_samples,).
            sample_rate (number, optional): Sampling rate of signal. If None, assumes
                                          self.sample_rate.

        Returns:
            torch.Tensor: The logmel feature vector of shape (batch_size, n_frames, n_mels)
                         or (n_frames, n_mels).
        """
        return self.extract(x, sample_rate)

    def extract(self, x, sample_rate=None):
        """Transform the given signal into a logmel feature vector.

        Args:
            x (torch.Tensor): Input time-series signal.
            sample_rate (number): Sampling rate of signal.

        Returns:
            torch.Tensor: The logmel feature vector.
        """
        # Ensure x is a tensor and add batch dimension if needed
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()

        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            unsqueezed = True
        else:
            unsqueezed = False

        # Resample if necessary
        if sample_rate is not None and sample_rate != self.sample_rate:
            self.resampler.orig_freq = sample_rate
            self.resampler.new_freq = self.sample_rate
            x = self.resampler(x)

        # Compute short-time Fourier transform
        stft = torch.stft(x,
                          n_fft=self.n_window,
                          hop_length=self.hop_length,
                          win_length=self.n_window,
                          window=torch.hann_window(self.n_window).to(x.device),
                          return_complex=True)

        # Compute magnitude spectrogram
        mag_spec = torch.abs(stft)

        # Apply Mel filterbank
        mel_spec = self.mel_scale(mag_spec)

        # Apply log nonlinearity (amplitude_to_db equivalent)
        # Using log10 with clipping to avoid log(0)
        log_mel = torch.log10(torch.clamp(mel_spec, min=1e-10))

        # Permute to (batch_size, n_frames, n_mels)
        log_mel = log_mel.permute(0, 2, 1)

        # Remove batch dimension if it was added
        if unsqueezed:
            log_mel = log_mel.squeeze(0)

        return log_mel


class MFCCExtractor(nn.Module):
    """Alternative feature extractor for MFCC features using PyTorch."""

    def __init__(self,
                 sample_rate=16000,
                 n_mfcc=13,
                 n_window=1024,
                 hop_length=512,
                 n_mels=64):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_window = n_window
        self.hop_length = hop_length

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_window,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )

        self.resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sample_rate)

    def output_shape(self, clip_duration):
        n_samples = int(clip_duration * self.sample_rate)
        n_frames = (n_samples - self.n_window) // self.hop_length + 1
        return (n_frames, self.n_mfcc)

    def extract(self, x, sample_rate=None):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()

        if x.dim() == 1:
            x = x.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False

        # Resample if necessary
        if sample_rate is not None and sample_rate != self.sample_rate:
            self.resampler.orig_freq = sample_rate
            self.resampler.new_freq = self.sample_rate
            x = self.resampler(x)

        mfcc = self.mfcc_transform(x)
        mfcc = mfcc.permute(0, 2, 1)  # (batch, frames, n_mfcc)

        if unsqueezed:
            mfcc = mfcc.squeeze(0)

        return mfcc

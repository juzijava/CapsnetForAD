import numpy as np
import torch
import torchaudio
import random


class AudioTransform:
    """Base class for audio transformations"""

    def __init__(self):
        pass

    def apply(self, audio, sample_rate):
        raise NotImplementedError


class PitchShift(AudioTransform):
    """Pitch shift transformation using PyTorch"""

    def __init__(self, n_samples=8, lower=-3.5, upper=3.5):
        super().__init__()
        self.n_samples = n_samples
        self.lower = lower
        self.upper = upper

    def apply(self, audio, sample_rate):
        """Apply pitch shift to audio"""
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio).float()

        # Generate random pitch shifts
        n_semitones = torch.linspace(self.lower, self.upper, self.n_samples)

        transformed = []
        for semitones in n_semitones:
            # Use torchaudio for pitch shifting
            effects = [
                ["pitch", f"{semitones.item()}"],  # Pitch shift in semitones
                ["rate", f"{sample_rate}"]
            ]

            # For simplicity, we'll use time stretching + resampling as approximation
            # Actual pitch shift implementation would be more complex
            shift_factor = 2 ** (semitones.item() / 12)
            n_samples = int(len(audio) / shift_factor)

            # Time stretching (simplified approach)
            if shift_factor > 1:
                # Higher pitch - shorten
                transformed_audio = audio[:n_samples]
            else:
                # Lower pitch - extend with zeros (simplified)
                transformed_audio = torch.nn.functional.interpolate(
                    audio.unsqueeze(0).unsqueeze(0),
                    size=n_samples,
                    mode='linear',
                    align_corners=False
                ).squeeze()

            transformed.append(transformed_audio)

        return transformed


class DynamicRangeCompression(AudioTransform):
    """Dynamic Range Compression using PyTorch"""

    def __init__(self, n_presets=2):
        super().__init__()
        self.n_presets = n_presets
        # Different compression ratios for different presets
        self.presets = [
            {'threshold': -20.0, 'ratio': 4.0, 'attack': 1.0, 'release': 50.0},  # 'radio'
            {'threshold': -30.0, 'ratio': 2.0, 'attack': 10.0, 'release': 100.0}  # 'film standard'
        ]

    def apply(self, audio, sample_rate):
        """Apply dynamic range compression"""
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio).float()

        transformed = []

        for i in range(min(self.n_presets, len(self.presets))):
            preset = self.presets[i]

            # Simplified DRC implementation
            # Convert to dB
            audio_db = 20 * torch.log10(torch.abs(audio) + 1e-8)

            # Apply compression curve
            compressed_db = torch.where(
                audio_db > preset['threshold'],
                preset['threshold'] + (audio_db - preset['threshold']) / preset['ratio'],
                audio_db
            )

            # Convert back to linear scale
            compressed_audio = 10 ** (compressed_db / 20) * torch.sign(audio)

            transformed.append(compressed_audio)

        return transformed


class Pipeline:
    """Pipeline for applying multiple transformations"""

    def __init__(self, steps):
        self.steps = steps

    def transform(self, audio, sample_rate):
        """Apply all transformations in the pipeline"""
        all_transformed = [audio]

        for name, transform in self.steps:
            if hasattr(transform, 'apply'):
                current_batch = []
                for audio_item in all_transformed:
                    transformed = transform.apply(audio_item, sample_rate)
                    current_batch.extend(transformed)
                all_transformed = current_batch

        return all_transformed


class Bypass:
    """Bypass wrapper for compatibility"""

    def __init__(self, deformer):
        self.deformer = deformer

    def apply(self, audio, sample_rate):
        return self.deformer.apply(audio, sample_rate)


def transformations(y, sample_rate, n_transforms):
    """Generate transformations for the given audio data.

    Args:
        y (torch.Tensor or np.ndarray): Input audio data.
        sample_rate (number): Sampling rate of audio.
        n_transforms (tuple): Number of transformations to apply.

    Yields:
        torch.Tensor: The transformed audio data.
    """
    # Return empty iterator if number of transforms is zero
    if n_transforms == (0, 0):
        return iter(())

    n_pitches, n_drc = n_transforms

    # Convert to tensor if needed
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    # Create deformers
    pitch_shifter = PitchShift(n_samples=n_pitches, lower=-3.5, upper=3.5)

    if n_drc > 0:
        drc = DynamicRangeCompression(n_presets=n_drc)
        # Create pipeline
        deformer = Pipeline(steps=[
            ('pitch_shift', Bypass(pitch_shifter)),
            ('drc', Bypass(drc))
        ])

        # Apply transformations
        transformed_audio = deformer.transform(y, sample_rate)
    else:
        # Only pitch shift
        transformed_audio = pitch_shifter.apply(y, sample_rate)

    # Return iterator
    return iter(transformed_audio)


def expand_metadata(metadata):
    """Duplicate the given metadata entries for data augmentation.

    Each metadata entry, which corresponds to a dataset example, is
    copied for every transformation that should be applied to the
    example. This is so that the new metadata structure reflects the
    augmented dataset. The copies are placed next to the original.

    Args:
        metadata (tuple): The metadata structure to expand.

    Returns:
        tuple: The expanded metadata structure.
    """
    names, target_values = metadata
    new_names, new_target_values = [], []

    for i, count in enumerate(transform_counts(target_values)):
        # Calculate number of copies (including original)
        n_copies = (count[0] + 1) * (count[1] + 1)
        for j in range(n_copies):
            new_names.append(f"{names[i]}_aug_{j}")
            new_target_values.append(target_values[i])

    return new_names, np.array(new_target_values)


def transform_counts(target_values):
    """Return a generator for the transformation counts of a dataset.

    Args:
        target_values (list): A list of target values for a dataset
            indicating which class each example belongs to.

    Yields:
        tuple: A tuple of the form ``(n_pitches, n_drc)``.
    """
    if len(target_values) == 0:
        return

    n_examples = np.sum(target_values, axis=0).astype(int)
    for y in target_values:
        # Determine how many transformations should be applied to this
        # example based on the smallest class it belongs to.
        relevant_classes = [label for label, value in enumerate(y) if value]
        if relevant_classes:
            min_n_examples = min(n_examples[label] for label in relevant_classes)
            yield transform_count(min_n_examples)
        else:
            yield (0, 0)


def transform_count(n_examples):
    """Return the number of transformations that should be applied to
    each example in a class.

    This function returns the number of pitch and dynamic range
    compression (DRC) transformations that should be applied to a class
    in which the total number of examples is equal to `n_examples`. The
    idea is that small classes should have a larger number of
    transformations applied in order to balance the dataset.

    Args:
        n_examples (int): The number of examples in the class.

    Returns:
        tuple: A tuple of the form ``(n_pitches, n_drc)``.
    """
    if n_examples < 500:
        return (8, 3)
    elif n_examples < 999:
        return (5, 2)
    elif n_examples < 4999:
        return (2, 1)
    elif n_examples < 9999:
        return (2, 0)

    return (0, 0)


# Alternative implementation using torchaudio's built-in transforms
class TorchAudioTransforms:
    """Alternative implementation using torchaudio transforms"""

    def __init__(self):
        self.pitch_shift = torchaudio.transforms.PitchShift(
            sample_rate=16000, n_steps=4
        )
        self.compressor = torchaudio.transforms.Vol(gain=0.5)

    def apply_pitch_shift(self, audio, n_shifts=8):
        """Apply multiple pitch shifts"""
        shifts = torch.linspace(-3.5, 3.5, n_shifts)
        transformed = []

        for shift in shifts:
            shifted = self.pitch_shift(audio.unsqueeze(0), shift.item())
            transformed.append(shifted.squeeze())

        return transformed


# Test function
def test_transformations():
    """Test the transformation functions"""
    # Create dummy audio
    sample_rate = 16000
    duration = 2.0
    audio = torch.randn(int(sample_rate * duration))

    # Test pitch shifting
    pitch_shifter = PitchShift(n_samples=3, lower=-2, upper=2)
    pitched = pitch_shifter.apply(audio, sample_rate)
    print(f"Pitch shift: {len(pitched)} variations")

    # Test DRC
    drc = DynamicRangeCompression(n_presets=2)
    compressed = drc.apply(audio, sample_rate)
    print(f"DRC: {len(compressed)} variations")

    # Test pipeline
    pipeline = Pipeline(steps=[
        ('pitch_shift', Bypass(pitch_shifter)),
        ('drc', Bypass(drc))
    ])

    final_transformed = pipeline.transform(audio, sample_rate)
    print(f"Pipeline: {len(final_transformed)} total variations")


if __name__ == "__main__":
    test_transformations()
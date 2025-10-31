import collections
import os.path

from . import training


work_path = 'D:/tools/Pycode/CapsnetForAD/workresult'
"""str: Path to parent directory containing program output."""

extraction_path = os.path.join(work_path, 'features')
"""str: Path to the directory containing extracted feature vectors."""

scaler_path = os.path.join(extraction_path, 'scaler.p')
"""str: Path to the scaler file used for standardization."""

model_path = os.path.join(work_path, 'models', training.training_id)
"""str: Path to the output directory of saved models."""

log_path = os.path.join(work_path, 'logs', training.training_id)
"""str: Path to the directory of TensorBoard logs."""

history_path = os.path.join(log_path, 'history.csv')
"""str: Path to log file for training history."""

predictions_path = os.path.join(
    work_path, 'predictions', training.training_id, '{}_{}_predictions.p')
"""str: Path to a model predictions file."""

results_path = os.path.join(
    work_path, 'results', training.training_id, '{}_{}_results.csv')
"""str: Path to the file containing results."""


Dataset = collections.namedtuple('Dataset',
                                 ['name',
                                  'path',
                                  'metadata_path',
                                  ])
"""Data structure encapsulating information about a dataset."""

_root_dataset_path = ('D:/tools/Pycode/CapsnetForAD/data')
"""str: Path to root directory containing input audio clips."""

training_set = Dataset(
    name='training',
    path=os.path.join(_root_dataset_path, 'audio'),
    metadata_path='data/metadata/training.csv',
)
"""Dataset instance for the training dataset."""

validation_set = Dataset(
    name='validation',
    path=os.path.join(_root_dataset_path, 'audio'),
    metadata_path='data/metadata/validation.csv',
)
"""Dataset instance for the validation dataset.

Note:
    The validation set is called the 'testing' set in DCASE 2017.
"""

test_set = Dataset(
    name='test',
    path=os.path.join(_root_dataset_path, 'audio'),
    metadata_path='data/metadata/test.csv',
)
"""Dataset instance for the test dataset.

Note:
    The test set is called the 'evaluation' set in DCASE 2017.
"""


def to_dataset(name):
    """Return the Dataset instance corresponding to the given name.

    Args:
        name (str): Name of dataset.

    Returns:
        The Dataset instance corresponding to the given name.
    """
    if name == 'training':
        return training_set
    elif name == 'validation':
        return validation_set
    elif name == 'test':
        return test_set
    return None

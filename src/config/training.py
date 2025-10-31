training_id = 'gccaps'
"""str: A string identifying this particular training configuration."""

initial_seed = 1000
"""int: Fixed seed used prior to training."""

batch_size = 16
"""int: The number of samples in a mini batch."""

n_epochs = 20
"""int: The number of epochs to train the network for.

A value of -1 indicates an early stopping condition should be used.
"""

learning_rate = {'initial': 0.0005,
                 'decay': 0.9,
                 'decay_rate': 2.,
                 }
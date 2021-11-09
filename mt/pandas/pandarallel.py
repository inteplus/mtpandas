'''Loading pandarallel.'''

from pandarallel import pandarallel
pandarallel.initialize(use_memory_fs=False)

loaded = True

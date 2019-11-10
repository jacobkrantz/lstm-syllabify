def try_tensorflow_import(verbose=False):
    """
    Sets the GPU device to 1. Sets the memory allocation to grow rather than
    allocating the whole VRAM.
    https://www.tensorflow.org/guide/gpu
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    import tensorflow as tf

    if verbose:
        tf.debugging.set_log_device_placement(
            True
        )  # logs what device is being used
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        return

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if verbose:
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

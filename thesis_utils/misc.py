import argparse
import os

def str2bool(v):
    """https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def remove_if_outputs_exists(file_dir, file):

    filepath = os.path.join(file_dir, file)
    if os.path.exists(filepath):
        os.unlink(filepath)
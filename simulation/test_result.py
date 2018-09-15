from simulation import *
import numpy as np
from generate_weights import *
from spectral_density import *
import matplotlib.pyplot as plt
import os
import pickle
import random
import matplotlib.pyplot as plt


def load_result(result_file_name = 'result'):
    print(os.path.join(RES_DIR, result_file_name))
    with open(os.path.join(RES_DIR, result_file_name), 'rb') as handle:
        res = pickle.load(handle)
    return res


res = load_result('ma_result_400')
print(res['ho_12']['precision'])
from preprocess import preprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


prep = preprocess()
prep.print_desc_stats()
prep.save_data('../../data/data_preprocessed.h5')
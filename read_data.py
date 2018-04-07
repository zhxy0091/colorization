import numpy as np
import os, sys, inspect
import random
from tensorflow.python.platform import gfile
import glob
import pandas as pd
import numpy as np
import _pickle as cPickle

utils_path = os.path.abspath(
	os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_path not in sys.path:
	sys.path.insert(0, utils_path)
import TensorflowUtils as utils

def read_dataset(data_dir):
	train_pickle_filename = "trainset-final.pickle"
	test_pickle_filename = "testset-final.pickle"
	train_pickle_filepath = os.path.join(data_dir, train_pickle_filename)
	test_pickle_filepath = os.path.join(data_dir, test_pickle_filename)
	if not os.path.exists(train_pickle_filepath):
		print("Cannot find train pickle file")
	else:
		print("Found train pickle file!")
	if not os.path.exists(test_pickle_filepath):
		print("Cannot find test pickle file")
	else:
		print("Found test pickle file!")

	train_data = pd.read_pickle(train_pickle_filepath)
	test_data = pd.read_pickle(test_pickle_filepath)


	print ("Training: %d,  Test: %d" % (
		len(train_data), len(test_data)))
	return train_data, test_data
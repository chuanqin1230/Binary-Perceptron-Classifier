import numpy as np
import pandas as pd

def read_data(datapath):
	# read form CSV
	train_df = pd.read_csv(datapath, header=0, index_col=0)
	test_df = pd.read_csv(datapath, header=0, index_col=0)
	
	# create training sets
	X_train = train_df.values
	X_test = test_df.values
	y_train = train_df.index
	y_test = test_df.index
	
	# create binary labels
	yb_train = []
	yb_test = []
	for label in train_df.index:
		if label % 2:
			yb_train.append(1)
		else:
			yb_train.append(-1)
	for label in test_df.index:
		if label % 2:
			yb_test.append(1)
		else:
			yb_test.append(-1)
	
	return X_train, X_test, yb_train, yb_test
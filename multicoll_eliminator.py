#!/usr/bin/env python3

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd
import numpy as np


def drop_highest_key(in_dict):
	"""
	Drops the key of the highest value in a dictionary
	in_dict:  a dictionary with keys and numeric values
	drop_key:  the key of the key/value pair that is dropped
	"""
	drop_key = max(in_dict.items(), key=operator.itemgetter(1))[0]
	del in_dict[drop_key]
	return in_dict, drop_key


def get_vifs(data):
	"""
	Calculates VIFs for a dataset.
	data:  dataframe (perhaps a numpy array would work) with the dataset you'd like to calculate VIFs for
	X:  your dataset with a constant added (needed for proper statsmodels VIF calculation)
	vif.to_dict():  a dictionary with each feature and its VIF
	"""
	X = add_constant(data)
	vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
	return X, vif.to_dict()


def eliminate_via_vif(data, undroppable = None, max_vif = 5):
	"""
	Recursively eliminates features from a dataset to combat multicollinearity
	data:  a pandas dataframe of your data
	undroppable:  any features that are undroppable
	max_vif:  a threshold above which no final feature's VIF may cross
	"""
	# add a constant to the undroppable feature list
	if 'const' not in undroppable:
		undroppable.append('const')

	# get features' VIFs, constant-added dataset
	X, vif_dict = get_vifs(data)

	# remove any infinite VIFs, except for those that belong to undroppable features
	infs = {k:v for k, v in vif_dict.items() if abs(v) == np.inf}
	left = {k:v for k, v in vif_dict.items() if k in undroppable or k not in list(infs.keys())}

	# Get your set of droppable candidate features
	droppable = {k:v for k, v in left.items() if v > max_vif and k not in undroppable}

	# if there are features above the VIF threshold, drop highest one and re-run
	if droppable != {}:
		features, dropped = drop_highest_key(droppable)
		new = X.copy(deep = True)
		new = new[[x for x in new.columns if x != dropped]]
		return eliminate_via_vif(new, undroppable = undroppable, max_vif = max_vif)

	# otherwise, returned the pared-down dataset
	else:
		left = {k:round(v, 3) for k, v in left.items()}
		final_features = [x for x in list(left.keys()) if x != 'const']
		return X[final_features]
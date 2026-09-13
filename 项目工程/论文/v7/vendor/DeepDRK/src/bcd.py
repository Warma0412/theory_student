"""
Implementation of a bayesian coefficient difference.
"""
import numpy as np
import knockpy
from knockpy import knockoff_stats as kstats

class BayesCoeffDiff(kstats.FeatureStatistic):

	def __init__(self, **kwargs):
		super().__init__()
		self.kwargs = kwargs

	def fit(
		self, 
		X, Xk, y,         
		groups=None,
		antisym="cd",
		group_agg="avg",
		cv_score=False,
		n_iter=2000,
		chains=5,
		**kwargs,
	):

		# Possibly set default groups
		n = X.shape[0]
		p = X.shape[1]
		if groups is None:
			groups = np.arange(1, p + 1, 1)
		features = np.concatenate([X, Xk], axis=1)

		# Handle kwargs
		for key in kwargs:
			self.kwargs[key] = kwargs[key]

		# Fit model
		import pyblip
		y_dist = kstats.parse_y_dist(y)
		if y_dist == 'gaussian':
			lm = pyblip.linear.LinearSpikeSlab(
				X=features, y=y, **self.kwargs
			)
		else:
			lm = pyblip.probit.ProbitSpikeSlab(
				X=features, y=y, **self.kwargs
			)
		lm.sample(N=n_iter, chains=chains)
		self.Z = np.mean(lm.betas, axis=0)
		self.groups = groups

		# Coefficient difference
		self.W = kstats.combine_Z_stats(self.Z, groups, antisym=antisym, group_agg=group_agg)
		return self.W

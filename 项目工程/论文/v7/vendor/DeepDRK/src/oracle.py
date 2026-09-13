import numpy as np
from knockpy import knockoff_stats as kstats

def oracle_fx_Wstat(
	beta,
	tildebeta,
	S
):
	# Calculate log odds
	eta = np.abs(tildebeta * beta) * np.diag(S)
	# Set signs
	psis = np.sign(beta)
	# Return W-statistics
	W = psis * eta * np.sign(tildebeta)
	return W

class OracleFXStatistic(kstats.FeatureStatistic):
	"""
	beta : np.ndarray
		``(p,)``-shaped estimate of linear coefficients
	lambd : float
		The regularization parameter to use while estimating beta.
	how_est : str
		How to estimate beta to use in the oracle statistic.
		One of "xi", "gd_cvx", "gd_noncvx", "em".
		This is ignored if beta is provided.
	"""

	def __init__(self, beta=None, how_est="gd_cvx", lambd=None):
		# Dummy attributes
		self.Z = None
		self.score = None
		self.score_type = None
		self.beta = beta
		self.lambd = lambd
		self.xi = None
		self.how_est = str(how_est).lower()

	def fit(
		self, X, Xk, groups, y, **kwargs
	):
		"""
		Fits the whitened oracle for FX knockoffs.
		Note that if beta is None, estimates beta.
		"""
		# Whitened estimator
		S = X.T @ X - X.T @ Xk
		Sinv = np.diag(1 / np.diag(S))
		self.tildebeta = np.dot(Sinv, np.dot(X.T - Xk.T, y))

		# Possibly estimate beta
		if self.beta is None:
			raise NotImplementedError("plug-in est not yet incorporated")
			# Method 1: cvxpy, possibly using the EM algorithm
			# if self.how_est == "xi" or self.how_est == "em":
			# 	self.only_xi = self.how_est == "xi"
			# 	self.hatbeta = self.penalized_mle(
			# 		X=X, Xk=Xk, y=y, S=S, **kwargs
			# 	)
			# # Method 2: gradient descent
			# elif self.how_est == "gd_cvx" or self.how_est == "gd_noncvx":
			# 	force_convex = self.how_est == 'gd_cvx'
			# 	self.solver = L1MLESolver(force_convex=force_convex)
			# 	self.hatbeta = self.solver.fit(
			# 		X=X, Xk=Xk, y=y, S=S, lambd=self.lambd, **kwargs
			# 	)
			# 	# Save some extraneous values
			# 	self.A = self.solver.A.detach().numpy()
			# 	self.xi = self.solver.xi.detach().numpy()
			# 	self.taus = self.solver.taus.detach().numpy()
			# else:
				# raise ValueError(f"Unrecognized estimation strategy {self.how_est}")
		else:
			self.hatbeta = None

		self.W = oracle_fx_Wstat(
			beta=self.hatbeta if self.beta is None else self.beta,
			tildebeta=self.tildebeta,
			S=S
		)
		return self.W

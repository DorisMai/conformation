"""
The :mod:`sklearn.kernel_approximation` module implements several
approximate kernel feature maps base on Fourier transforms.
"""

# Author: Andreas Mueller <amueller@ais.uni-bonn.de>
#
# License: BSD 3 clause

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array, check_random_state, as_float_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels
import collections



class Nystroem(BaseEstimator, TransformerMixin):
	"""Approximate a kernel map using a subset of the training data.

	Constructs an approximate feature map for an arbitrary kernel
	using a subset of the data as basis.

	Parameters
	----------
	kernel : string or callable, default="rbf"
	    Kernel map to be approximated. A callable should accept two arguments
	    and the keyword arguments passed to this object as kernel_params, and
	    should return a floating point number.

	n_components : int
	    Number of features to construct.
	    How many data points will be used to construct the mapping.

	gamma : float, default=None
	    Gamma parameter for the RBF, polynomial, exponential chi2 and
	    sigmoid kernels. Interpretation of the default value is left to
	    the kernel; see the documentation for sklearn.metrics.pairwise.
	    Ignored by other kernels.

	degree : float, default=3
	    Degree of the polynomial kernel. Ignored by other kernels.

	coef0 : float, default=1
	    Zero coefficient for polynomial and sigmoid kernels.
	    Ignored by other kernels.

	kernel_params : mapping of string to any, optional
	    Additional parameters (keyword arguments) for kernel function passed
	    as callable object.

	random_state : {int, RandomState}, optional
	    If int, random_state is the seed used by the random number generator;
	    if RandomState instance, random_state is the random number generator.


	Attributes
	----------
	components_ : array, shape (n_components, n_features)
	    Subset of training points used to construct the feature map.

	component_indices_ : array, shape (n_components)
	    Indices of ``components_`` in the training set.

	normalization_ : array, shape (n_components, n_components)
	    Normalization matrix needed for embedding.
	    Square root of the kernel matrix on ``components_``.


	References
	----------
	* Williams, C.K.I. and Seeger, M.
	  "Using the Nystroem method to speed up kernel machines",
	  Advances in neural information processing systems 2001

	* T. Yang, Y. Li, M. Mahdavi, R. Jin and Z. Zhou
	  "Nystroem Method vs Random Fourier Features: A Theoretical and Empirical
	  Comparison",
	  Advances in Neural Information Processing Systems 2012


	See also
	--------
	RBFSampler : An approximation to the RBF kernel using random Fourier
	             features.

	sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.
	"""
	def __init__(self, basis = None, kernel="rbf", gamma=None, coef0=1, degree=3, kernel_params=None, n_components=100, random_state=None):
		self.kernel = kernel
		self.gamma = gamma
		self.coef0 = coef0
		self.degree = degree
		self.kernel_params = kernel_params
		self.n_components = n_components
		self.random_state = random_state
		self.basis = basis

	def fit(self, X, y=None):
		"""Fit estimator to data.

		Samples a subset of training points, computes kernel
		on these and computes normalization matrix.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_feature)
		    Training data.
		"""
		X = check_array(X, accept_sparse='csr')
		rnd = check_random_state(self.random_state)
		n_samples = X.shape[0]

		# get basis vectors
		basis = self.basis
		print("Here's what goes into landmarknystroem:")
		print((np.shape(X)))
		print((np.shape(basis)))
		if basis is not None:
			if self.n_components > n_samples:
				# XXX should we just bail?
				n_components = n_samples

			else:
				n_components = self.n_components
			n_components = min(n_samples, n_components)
			inds = rnd.permutation(n_samples)
			basis_inds = inds[:n_components]
			basis = X[basis_inds]
			self.components_ = basis
			self.component_indices_ = inds
		else:
			n_components = self.n_components
			self.components_ = basis

		basis_kernel = pairwise_kernels(basis, metric=self.kernel, filter_params=True,**self._get_kernel_params())

		# sqrt of kernel matrix on basis vectors
		U, S, V = svd(basis_kernel)
		S = np.maximum(S, 1e-12)
		self.normalization_ = np.dot(U * 1. / np.sqrt(S), V)

		return self

	def transform(self, X):
	    """Apply feature map to X.

	    Computes an approximate feature map using the kernel
	    between some training points and X.

	    Parameters
	    ----------
	    X : array-like, shape=(n_samples, n_features)
	        Data to transform.

	    Returns
	    -------
	    X_transformed : array, shape=(n_samples, n_components)
	        Transformed data.
	    """
	    check_is_fitted(self, 'components_')
	    X = check_array(X, accept_sparse='csr')

	    kernel_params = self._get_kernel_params()
	    embedded = pairwise_kernels(X, self.components_,
	                                metric=self.kernel,
	                                filter_params=True,
	                                **kernel_params)
	    return np.dot(embedded, self.normalization_.T)

	def _get_kernel_params(self):
	    params = self.kernel_params
	    if params is None:
	        params = {}
	    if not isinstance(self.kernel, collections.Callable):
	        params['gamma'] = self.gamma
	        params['degree'] = self.degree
	        params['coef0'] = self.coef0

	    return params
from sklearn.base import TransformerMixin
from sklearn.utils import check_array



class MyExtractor(TransformerMixin):
    
	def __init__(self, func, *featurizers):
		self.analyzer = func
		self.featurizers = featurizers

	def fit(self,X,y=None):
		return self

	def transform(self,X):
		out = [[self.analyzer(val)] for val in X]
		out = check_array(out)
		return out
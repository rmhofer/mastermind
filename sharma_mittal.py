#sharma_mittal.py
import numpy as np

def sm_entropy(prob, t, r):
	# print t, r
	"""
		function to compute Sharma-Mittal entropy
		for any probability distribution 'prob' 
		and order-degree parameter pair
	"""
	#convert to numpy array and flatten
	prob = np.array(prob).flatten()
	prob = np.array([p for p in prob if p != 0])

	absErrorTolerance = 1e-15
	#values must be non-negative
	# for p, i in enumerate(prob):
	# 	 if abs(p) <= absErrorTolerance: 
	# 	 	prob[i] = 0
	# 	 if p < 0:
	# 	 	print p
	# 	 	raise ValueError('values must be positive!')

	#re-normalize
	prob = prob/np.sum(prob)
	
	if (np.sum(prob) - 1.0) >= absErrorTolerance:
		raise ValueError('Probabilities do not sum to 1')
	
	r_threshold = 255
	t_threshold = 255

	#catch exceptions
	for i, p in enumerate(prob):
		if p < 0.0 or p > 1.0:
			# print "P:", p
			prob[i] = np.max([0.0, prob[i]])
			prob[i] = np.min([1.0, prob[i]])
			# raise ValueError('Probabilities not between 0 and 1')
	
	# r or t = inf.
	if r >= 257: r = r_threshold
	if t >= 257: t = t_threshold

	#Shannon entropy
	if r==1.0 and t==1.0: 
		# print "Shannon"
		return np.sum([p*np.log(1/p) 
			for p in prob if not p==0])
	#Gaussian
	elif r==1.0:
		# print "Gaussian"
		return (1-np.exp(((1-t)*np.sum([p*np.log(1/p) 
			for p in prob if not p==0]))))/(t-1)
	#Renyi
	elif t==1.0:
		return (np.log(np.sum([np.power(p,r) 
			for p in prob])))/(1-r)

	#Limits
	elif r==0.0:
		return np.log(np.sum([0 if p==0.0 else 1 
			for p in prob]))
	elif r==r_threshold:
		return np.log(1/np.amax(prob))
	
	#Effective number
	elif t==0.0:
		return np.power(np.sum([np.power(p,r) 
			for p in prob]),1/(1-r))-1
	
	#Generic sharma-mittal formula
	else:
		return (1-np.power(np.sum([np.power(p,r) 
			for p in prob]),(t-1)/(r-1)))/(t-1)
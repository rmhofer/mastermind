#Mastermind.py
from __future__ import division
import numpy as np
import sharma_mittal
# import seaborn as sns
# import matplotlib.pyplot as plt
from fractions import Fraction

########################################################################
class Game:
	'''

	'''
	#----------------------------------------------------------------------
	def __init__(self, **kwargs):
		''' A Mastermind problem is characterized by codelength
			and number of colors

			### Attributes
			- codelength: number of pegs
			- codejar: pass on frequency list to sample code from 
						codejar
			- maxguess: maximum number of guesses

		'''

		self.colors = []
		self.logging = kwargs.get('logging', False)
		self.codelength = int(kwargs.get('codelength', 4))
		self.codejar = list(kwargs.get('codejar', False))
		self.maxguess = int(kwargs.get('maxguess', 12))
		if not isinstance(self.codejar, list):
			self.Ncolors = int(kwargs.get('Ncolors', 6))
			self.codejar = np.ones(self.Ncolors)
		else: self.Ncolors = len(self.codejar)		
		self.prior = np.array(self.codejar)/np.sum(self.codejar)
	
	#----------------------------------------------------------------------
	def initialize(self, **kwargs):
		self.code = kwargs.get('code', False) #check if valid
		if not self.code:
			self.code = np.random.choice(
				self.Ncolors, size=self.codelength, 
				replace=True, p=self.prior) + 1
		self.code = np.array(self.code)
		if self.logging:
			print "true code: %s" % str(self.code)
			print "+---------+---------+"
		self.step = 0
		self.combinations = np.zeros((0, self.codelength))
		self.feedbacks = []
		self.codepool = self.get_feasible_set()
		self.end = False
		self.currentFS = [self.codepool, self.fs_probability(self.codepool), self.step]
	
	#----------------------------------------------------------------------	
	def getCurrentFS(self):
		if self.currentFS[2] != self.step:
			uFS = self.update_feasible_set(self.currentFS[0])
			self.currentFS = np.array([uFS, self.fs_probability(uFS), self.step])
		return self.currentFS

	#----------------------------------------------------------------------
	def viscodejar(self, ax=None):
		if ax == None: f, ax = plt.subplots(1)
		sns.barplot(np.arange(self.Ncolors), self.codejar, palette="Set3", ax=ax)
		plt.show()
		return ax

	#----------------------------------------------------------------------
	def fs_probability(self, fs):
		probs = []
		for c in fs: probs.append(self.get_probability(c))
		probs /= np.sum(probs)
		return probs

	#----------------------------------------------------------------------
	def fs_entropy(self, fs, t, r):
		probs = self.fs_probability(fs)
		return sharma_mittal.sm_entropy(probs, t=t, r=r)

	#----------------------------------------------------------------------
	def visualize_query(self, c, feasible_set, ax=None):
		if ax == None: f, ax = plt.subplots()
		cmap = sns.cubehelix_palette(8, start=.5, rot=-.75, 
			light=.98, as_cmap=True, reverse=True)
		N = 10
		space = np.logspace(-4, N-4, num=N, endpoint=False, 
			base=2.0, dtype=np.float64)
		space[0] = 0
		evarr = np.zeros((N,N))

		for (i,j), u in np.ndenumerate(evarr):
			prior_ent = self.fs_entropy(
				feasible_set, t=space[i], r=space[j]) 
			post_ent = self.evaluate_combination(
				c, feasible_set, t=space[i], r=space[j])
			# print (prior_ent - post_ent)/post_ent
			evarr[i][j] = (prior_ent - post_ent)/post_ent

		labels = [('%s' % Fraction(f)) for f in space]
		ax = sns.heatmap(evarr, ax=ax, cbar=False, cmap=cmap,
			xticklabels=labels, yticklabels=labels)
		ax.set(xlabel='Order (r)', ylabel='Degree (t)')
		ax.set_aspect('equal')
		ax.invert_yaxis()
		plt.show()
		return ax

	#----------------------------------------------------------------------
	def compute_console_statistics(self):
		self.getCurrentFS()

		#------ compute position statistics
		position_statistics = np.zeros(shape=(self.Ncolors, self.codelength))
		for p in np.arange(self.codelength):
			position_vector = self.currentFS[0][:,p]
			for c in np.arange(self.Ncolors):
				idx = np.where(position_vector==(c+1))
				position_statistics[c][p] += np.sum(self.currentFS[1][idx])
		
		#------ compute color statistics
		color_statistics = np.zeros(shape=(self.Ncolors, self.codelength+1))
		for code, p in zip(self.currentFS[0], self.currentFS[1]):
			count = np.bincount(code)[1:]
			count = np.pad(count, (0, self.Ncolors-len(count)),
				mode='constant', constant_values=0)
			for color, number in np.ndenumerate(count):
				color_statistics[color][number] += p

		#------ compute fs statistics
		fs_size = len(self.currentFS[0])
		fs_entropy = sharma_mittal.sm_entropy(
			self.currentFS[1], t=1.0, r=1.0)
		n = 5
		topIDX = [np.argsort(self.currentFS[1])][0][-n:]
		topC = self.currentFS[0][topIDX][::-1]
		topP = self.currentFS[1][topIDX][::-1]
		fs_stats = [fs_size, fs_entropy, topC, topP]
		# 
		return [position_statistics, color_statistics, fs_stats]
		

	#----------------------------------------------------------------------
	def partition(self, feasible_set):
		''' compute partition matrix for feasible set. 
			The information in the matrix can be considered
			a lookahead table, but is expensive to compute and 
			requires the feasible set!
		'''
		partition_matrix = np.zeros(shape=(len(feasible_set),
			self.codelength+1,self.codelength+1))
		for i, c_i in enumerate(feasible_set):
			for j, c_j in enumerate(feasible_set):
				r = self.response(c_i, c_j)
				b = r['position']
				w = r['color']
				partition_matrix[i][b][w] += 1 
				partition_matrix[j][b][w] += 1 
		# print partition_matrix[::]
		return partition_matrix

	#----------------------------------------------------------------------
	def get_probability(self, combination):
		#probability that an item is the hidden code!
		prob = 1
		for i in np.arange(self.codelength):
			prob *= self.prior[combination[i]-1]
		return prob
	
	#----------------------------------------------------------------------
	def get_random(self):
		''' function to generate a random combination '''
		return np.random.choice(
				self.Ncolors, size=self.codelength, 
				replace=True, p=self.prior) + 1

	#----------------------------------------------------------------------
	def update_feasible_set(self, feasible_set):
		new_feasible_set = np.zeros((0, self.codelength),dtype=int)
		for combination in feasible_set:
			if self.feasible(combination):
				new_feasible_set = np.vstack((new_feasible_set, combination))
		return new_feasible_set

	#----------------------------------------------------------------------
	def get_feasible_set(self):
		feasible_set = np.zeros((0, self.codelength),dtype=int)
		for index, x in np.ndenumerate(
			np.empty(shape=[self.Ncolors] * self.codelength)):
			combination = np.array(index) + 1
			if self.feasible(combination):
				feasible_set = np.vstack((feasible_set, combination))
		return feasible_set

	#----------------------------------------------------------------------
	def consistent(self, c, combination):
		''' the true code is a subset of all consistent combinations '''
		return (self.response(c, combination) 
			== self.response(combination, self.code))

	#----------------------------------------------------------------------
	def feasible(self, c):
		''' a combinaiton is feasible if it is consistent with all 
			combinations played so far '''
		for played_combination in self.combinations:
			if not self.consistent(c, played_combination):
				return False
		return True

	#----------------------------------------------------------------------
	def evaluate_combination(self, target, feasible_set, t, r, logging=False):
		#target: combination to be evaluated
		if logging: print "\n### Evaluating combination: ", target
		collected_responses = []
		probabilities = np.zeros(self.codelength**2)
		for combination in feasible_set:
			response = self.response(target, combination)
			probability = self.get_probability(combination)
			if response not in collected_responses:
				collected_responses.append(response)
			idx = np.where(np.array(collected_responses)==response)[0]
			probabilities[idx] += probability
		probabilities = (probabilities[:len(collected_responses)] 
			/ np.sum(probabilities))
		collected_responses = np.array(collected_responses)

		prior_entropy = self.fs_entropy(feasible_set, t, r)

		p_f_sets = []
		for response in collected_responses:
			f_rc = []
			if logging:  print "\n -> when response is: ", response
			for combination in feasible_set:
				if (self.response(combination, target) == response):
					f_rc.append(combination)
			tmp = np.array([self.get_probability(c) for c in f_rc])
			tmp /= np.sum(tmp)
			if logging: print tmp
			tmp =  sharma_mittal.sm_entropy(tmp, t=t, r=r)
			p_f_sets.append(tmp)
		p_f_sets = np.array(p_f_sets)
		exp_post_entropy = np.sum(np.multiply(p_f_sets, probabilities))
		return prior_entropy - exp_post_entropy

	#----------------------------------------------------------------------
	def best_combination(self, feasible_set, t, r):
		#compute information gain of all combinations in the codepool:
		evaluations = [self.evaluate_combination(combination, 
			feasible_set, t, r) for combination in self.codepool]
		idx = np.argwhere(evaluations == np.amax(evaluations))
		queries = self.codepool[idx.flatten()]
		if self.logging:
			print "\t --> u(best query for oder[r]=%s, degree[t]=%s): %.4f" % (
				r, t, np.amax(evaluations))
			print "\t --> equivalence class of best queries:\n%s" % queries
		return queries

	#----------------------------------------------------------------------
	def best_mixes(self, t, r, p):
		#check if p is between 0 and 1!!!
		evaluations = np.array([self.evaluate_combination(combination, 
			self.currentFS[0], t, r) for combination in self.codepool])
		evaluations *= float(1.0-p)
		print "sclaed information: \n", np.array(evaluations)
		
		probArray = []
		for combination in self.codepool:
			if self.feasible(combination):
				probArray.append(self.get_probability(combination))
			else:
				probArray.append(0)
		probArray = np.array(probArray) / np.sum(probArray)
		probArray *= float(p)

		print "scaled probabilities \n", probArray
		combination = np.add(evaluations, probArray)
		print "scaled combination\n", combination

		idx = np.argwhere(combination == np.amax(combination))
		queries = self.codepool[idx.flatten()]
		return queries

	#----------------------------------------------------------------------
	def response(self, combination, code):
		''' simulate a response given a combinationb and a(!) true code '''
		if (not len(combination) == self.codelength or 
			not (np.array(combination) <= self.Ncolors).all()):
			raise ValueError('Combination not valid!')
		combination = np.array(combination)
		feedback = {'position' : 0, 'color' : 0}
		exclude = []
		for i in np.arange(self.codelength):
			if combination[i] == code[i]:
				feedback['position'] += 1
				exclude.append(i)
		for i in np.delete(np.arange(self.codelength), exclude):
			if combination[i] in np.delete(code, exclude):
				feedback['color'] += 1
				exclude.append(np.setdiff1d(
					np.where(code==combination[i])[0], exclude)[0])
		return feedback

	#----------------------------------------------------------------------
	def guess(self, combination):
		feedback = self.response(combination, self.code)
		combination = np.array(combination)
		self.combinations = np.vstack((
			self.combinations, combination))
		self.feedbacks.append(feedback)
		self.step += 1
		if self.step == self.maxguess:
			print "Max number of guesses reached!!"
		if self.logging: print "%s. guess:  %s \t\t" \
			"feedback: %s" % (self.step, 
				str(combination), feedback)
		if np.array_equal(self.code, combination):
			self.end = True
			if self.logging: print "+---------+---------+\n\n\n"
		return feedback

'''
class AutomatizedPlayAgent():
	def __init__(self, **kwargs):
		self.game = Game(**kwargs)
		self.args = kwargs
	def random_play(self):
		# random search 
		g = self.game
		g.initialize(**self.args)

		while not g.end:
			c = 0
			while True:
				c = g.get_random()
				if g.feasible(c): break
			f = g.guess(c)
		return g.step
	
	def knuth_play(self):
		# pick based on partition matrix
		g = self.game
		g.initialize(**self.args)
		g.guess(g.get_random())

		while not g.end:
			fs = g.get_feasible_set()
			pa = g.partition(fs)
			idx = np.argmin([np.max(pa[i,:]) 
				for i in np.arange(fs.shape[0])])
			# idx = np.unravel_index(pa.argmax(), pa.shape)[0]
			c = np.array(fs[idx], dtype=int)
			f = g.guess(c)
		return g.step

	def entropy_play(self, t, r):
		g = self.game
		g.initialize(**self.args)
		fs = g.codepool
		while not g.end:
			if fs.shape[0]<=2:
				probabilities = [g.get_probability(c) for c in fs]
				f = g.guess(fs[np.argmax(probabilities)])
			else:
				bc = g.best_combination(fs, t=t, r=r)
				f = g.guess(bc[np.random.randint(bc.shape[0])])
			fs = g.update_feasible_set(fs)
			# fs = g.get_feasible_set()
		return g.step
'''

########################################################################
class AppAgent():
	''' 

	'''
	#----------------------------------------------------------------------
	def __init__(self, **kwargs):
		self.game = kwargs.get('game', False)
		self.mode = kwargs.get('mode', 1)
		self.r = kwargs.get('r', 1)
		self.t = kwargs.get('t', 1)
		self.p = kwargs.get('p', 0.5) #mix parameter

		if self.game == False:
			self.game = Game()
			self.game.initialize()

		#dictionary with mode -> algorithm mappings
		stratDict = {
			1 : self.random_play,
			2 : self.random_play,  #knuth_play
			3 : self.pure_probability,
			4 : self.pure_information_gain,
			5 : self.mixed_strategy 
		}

		self.strategy = stratDict[self.mode]
		self.fs = []

	def compute_guess(self):	
		return self.strategy()

	#mode = 1: random play
	#----------------------------------------------------------------------
	def random_play(self):
		while True:
			c = self.game.get_random()
			if self.game.feasible(c): break
		return c

	#mode = 2: knuth play
	#----------------------------------------------------------------------
	def knuth_play(self):
		return self.game.get_random()

	#mode = 3: select most probable
	#----------------------------------------------------------------------
	def pure_probability(self):
		self.game.getCurrentFS()
		'''
		maxProbIDXs = np.argmax(self.game.currentFS[1]) #first occurence
		maxProbIDXs = np.where(self.game.currentFS[1]==self.game.currentFS[1][maxProbIDXs])[0]
	
		#chose guess at random from set of max prob guesses
		if len(maxProbIDXs) > 1:
			return self.game.currentFS[0][maxProbIDXs[np.random.randint(len(maxProbIDXs))]]
		else:
			return self.game.currentFS[0][maxProbIDXs]
		'''

		n = 5
		topIDX = [np.argsort(self.game.currentFS[1])][0][-n:]
		topC = self.game.currentFS[0][topIDX][::-1]
		return topC[0]

	#mode = 4: sharma mittal
	#----------------------------------------------------------------------
	def pure_information_gain(self):
		self.game.getCurrentFS()
		cs = self.game.best_combination(self.game.currentFS[0], t=self.t, r=self.r)
		
		if len(self.game.currentFS[0]) == 1:
			print self.game.currentFS[0][0]
			return self.game.currentFS[0][0]
		#chose guess at random from set of max info gain guesses
		if cs.shape[0] > 1:
			return cs[np.random.randint(cs.shape[0])]
		else:
			return cs
		
	def mixed_strategy(self):
		self.game.getCurrentFS()
		cs = self.game.best_mixes(t=self.t, r=self.r, p=self.p)
		
		if len(cs) == 1:
			return cs[0] 

		if len(self.game.currentFS[0]) == 1:
			print self.game.currentFS[0][0]
			return self.game.currentFS[0][0]
		#chose guess at random from set of max info gain guesses
		if cs.shape[0] > 1:
			return cs[np.random.randint(cs.shape[0])]
		else:
			return cs

		# probabilities = np.zeros(len(cs))
		# for c in cs:
		# 	for i, fs_c in enumerate(self.game.currentFS[0]):
		# 		if list(c) == list(fs_c):
		# 			print i
		# 			print self.game.currentFS[1][i]
			# print np.where(self.game.currentFS[0]==[c])
		# print probabilities
		# if self.fs.shape[0]<=2:
		# 	probabilities = [self.game.get_probability(c) for c in self.fs]
		# 	return self.fs[np.argmax(probabilities)]
		# else:
		# 	bc = self.game.best_combination(self.fs, t=t, r=r)
		# 	return bc[np.random.randint(bc.shape[0])]



# dist1 = [0.5, 0.3, 0.1, 0.05, 0.025, 0.025]
# dist2 = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
# dist3 = [0.4, 0.4, 0.2]
# dist = [dist1, dist2, dist3]

# #1 = shannon
# #2 = prob gain
# rarr = [1,5,2]
# tarr = [1,1.8,2]

# for r,t in zip(rarr, tarr):
# 	ent = [sharma_mittal.sm_entropy(d, r=r, t=t) for d in dist]
# 	print np.array(ent)


# d = [0.94, 0.06]
# d = [0.63, 0.27, 0.1]
# d = [0.37, 0.41, 0.22]
# d = [0.50, 0.50]
# d = [0.88, 0.12]
d = [0.0, 0.0]
print sharma_mittal.sm_entropy(d, r=1.0, t=1.0)



# r=1.0
# t=1.0
# print sharma_mittal.sm_entropy([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], r=r, t=t)
# print sharma_mittal.sm_entropy([0.4, 0.3, 0.2, 0.1], r=r, t=t)


# print np.random.choice(4, size=4, replace=True, p=[0.4, 0.3, 0.2, 0.1])

###############
# # GAME
# ###############
# game = Game(codelength=3, codejar=[8,6,6,2], logging=True)
# game.initialize()
# game.guess(combination = [1,3,2])
# game.guess(combination = [4,1,2])

# agent = AppAgent(game=game, mode=5)
# print agent.compute_guess()

# game.compute_console_statistics()


# fs = game.get_feasible_set()		

# print p

# game.visualize_query([1,1,4], fs, ax=None)

# # game.viscodejar()
# 			#initialize game
# # game.guess(combination = [1,1,2,3])	#make first guess
# # game.guess(combination = [3,4,4,4])	#make first guess

# bc = game.best_combination(fs, t=1, r=1)
# print bc[np.random.randint(bc.shape[0])]

# print game.evaluate_combination([2,4,2], fs, t=1, r=1)
# print game.evaluate_combination([1,2,3], fs, t=1, r=1)
# print game.evaluate_combination([1,2,2,4], fs, t=1, r=1)
# print game.evaluate_combination([2,4,3,4], fs, t=1, r=1)

###############
# AGENT
###############
# a = Agent(codelength=3, codejar=[1,1,1,1], logging=True)

# a.entropy_play(r=128, t=2)
# a.entropy_play(r=1, t=1)
# a.entropy_play(r=5, t=1.8)

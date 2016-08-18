"""
	mastermind.py

	This is the python module that provides the main (backbone)
	funcitonality, i.e., the program logics, for the mastermind
	GUI-based application

	It can also be used to run simulations independently of 
	using the GUI (see blow)

"""

from __future__ import division
import numpy as np
import sharma_mittal
from fractions import Fraction


"""
	Part one: the game

	Simply consists of functions to play the mastermind game
	in the code jar variation
"""


########################################################################
class Game:
	"""
		Main game class. Creat an instance and initialize various
		attribues (e.g., code length, code jar). Can be used 
		repeatedly to play several mastermind games 

	"""
	#----------------------------------------------------------------------
	def __init__(self, **kwargs):
		"""
			The attributes are:
			- logging: if logging output in the console is wanted
			- codelength: number of pegs/marbles
			- codejar: pass on frequency list that code is sampled from
			- maxguess: maximum number of guesses before game ends
		"""
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
		""" Initialize game and compute set of all feasible codes """
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
		""" 
			function returns the current feasible set
			array is either read out from memory if it exists, or updated
			first and then returned if it is not up to date

		"""
		if self.currentFS[2] != self.step:
			uFS = self.update_feasible_set(self.currentFS[0])
			self.currentFS = np.array([uFS, self.fs_probability(uFS), self.step])
		return self.currentFS

	#----------------------------------------------------------------------
	def viscodejar(self, ax=None):
		"""
			Visualize the codejar as a historgram

			If used, add the following lines to program preamble:
	 		- import seaborn as sns
			- import matplotlib.pyplot as plt
		"""

		if ax == None: f, ax = plt.subplots(1)
		sns.barplot(np.arange(self.Ncolors), self.codejar, palette="Set3", ax=ax)
		plt.show()
		return ax

	#----------------------------------------------------------------------
	def fs_probability(self, fs):
		""" returns probability of all items in the feasible set """

		probs = []
		for c in fs: probs.append(self.get_probability(c))
		probs /= np.sum(probs)
		return probs

	#----------------------------------------------------------------------
	def fs_entropy(self, fs, t, r):
		""" returns Sharma-Mittal entropy of feasible """

		probs = self.fs_probability(fs)
		return sharma_mittal.sm_entropy(probs, t=t, r=r)

	#----------------------------------------------------------------------
	def visualize_query(self, c, feasible_set, ax=None):
		"""
			Heatmap to show information gain for a specific query 
			across the Sharma-Mittal space

			If used, add the following lines to program preamble:
	 		- import seaborn as sns
			- import matplotlib.pyplot as plt
		"""

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
		"""
			Funciton to compute (position/color/feasible set) statistics
			used by the MMind_App. The statistics are displayed in the
			statistics widget
		"""

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

		return [position_statistics, color_statistics, fs_stats]
		

	#----------------------------------------------------------------------
	def partition(self, feasible_set):
		''' 
			compute partition matrix for feasible set. 
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
		return partition_matrix

	#----------------------------------------------------------------------
	def get_probability(self, combination):
		""" 
			return the probability that 'combination' 
			is the hidden code 
		"""

		prob = 1
		for i in np.arange(self.codelength):
			prob *= self.prior[combination[i]-1]
		return prob
	
	#----------------------------------------------------------------------
	def get_random(self):
		""" use this funciton to generate a random combination """

		return np.random.choice(
				self.Ncolors, size=self.codelength, 
				replace=True, p=self.prior) + 1

	#----------------------------------------------------------------------
	def update_feasible_set(self, feasible_set):
		""" update feasible set if called """

		new_feasible_set = np.zeros((0, self.codelength),dtype=int)
		for combination in feasible_set:
			if self.feasible(combination):
				new_feasible_set = np.vstack((new_feasible_set, combination))
		return new_feasible_set

	#----------------------------------------------------------------------
	def get_feasible_set(self):
		""" function to compute the feasible set (initially) """
		feasible_set = np.zeros((0, self.codelength),dtype=int)
		for index, x in np.ndenumerate(
			np.empty(shape=[self.Ncolors] * self.codelength)):
			combination = np.array(index) + 1
			if self.feasible(combination):
				feasible_set = np.vstack((feasible_set, combination))
		return feasible_set

	#----------------------------------------------------------------------
	def consistent(self, c, combination):
		""" check consistency of two combinations passed as arguments """		
		return (self.response(c, combination) 
			== self.response(combination, self.code))

	#----------------------------------------------------------------------
	def feasible(self, c):
		""" 
			a combinaiton is feasible if it is consistent with all 
			combinations played so far 
		"""
		for played_combination in self.combinations:
			if not self.consistent(c, played_combination):
				return False
		return True

	#----------------------------------------------------------------------
	def evaluate_combination(self, target, feasible_set, t, r, logging=False):
		"""
			return expected information gain of target combination (target)
			for a specific degree-order pair

		"""
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
		""" 
			find combination with highest expected information
			gain (for specific degree-order pair) in the entire
			code pool (all possible codes)
		"""

		evaluations = [self.evaluate_combination(combination, 
			feasible_set, t, r) for combination in self.codepool]

		idx = np.argwhere(abs(evaluations - 
			np.amax(evaluations)) <= 1e-10)

		queries = self.codepool[idx.flatten()]
		if self.logging:
			print "\t --> u(best query for oder[r]=%s, degree[t]=%s): %.4f" % (
				r, t, np.amax(evaluations))
			print "\t --> equivalence class of best queries:\n%s" % queries
		return queries

	#----------------------------------------------------------------------
	def best_mixes(self, t, r, p):
		"""
			return the best 'mixed' combination
		"""

		evaluations = np.array([self.evaluate_combination(combination, 
			self.currentFS[0], t, r) for combination in self.codepool])
		evaluations *= 2 * float(1.0-p)
		# print "scaled information: \n", np.array(evaluations)
		
		probArray = []
		for combination in self.codepool:
			if self.feasible(combination):
				probArray.append(self.get_probability(combination))
			else:
				probArray.append(0)
		probArray = np.array(probArray) / np.sum(probArray)
		probArray *= 2 * float(p)
		# print "scaled probabilities \n", probArray
		combined = np.add(evaluations, probArray)
		# print "scaled combination\n", combined

		idx = np.argwhere(abs(combined - 
			np.amax(combined)) <= 1e-10)
		queries = self.codepool[idx.flatten()]
		return queries

	#----------------------------------------------------------------------
	def response(self, combination, code):
		""" 
			simulate response for combination 'combination' if 
			'code' were the true hidden code
		"""
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
		"""
			guess combination (combination) and return feedback
		"""
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



"""
	Part two: playing agents

	Consists of functions for artificial agents to play the
	game using different stratgies
"""


########################################################################
class AppAgent():
	""" 
		Create an AppAgent instance and choose which 
		strategy it should follow
	"""
	#----------------------------------------------------------------------
	def __init__(self, **kwargs):
		"""	
			- use an already existing game by passing on a GAME instance
				for game
			- mode, an integer, determines which strategy is used 
				(see below)
			- r, t for Sharma-Mittal, p is the mixing parameter
		"""
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
			2 : self.pure_probability,
			3 : self.pure_information_gain,
			4 : self.mixed_strategy 
		}

		self.strategy = stratDict[self.mode]
		self.fs = []

	#----------------------------------------------------------------------
	def compute_guess(self):
		""" perform a guess given the strategy chosen for the agent """
		return self.strategy()
	
	#----------------------------------------------------------------------
	def play(self):
		while game.end == False:
			game.guess(combination=agent.compute_guess())

	#mode = 1: random play
	#----------------------------------------------------------------------
	def random_play(self):
		while True:
			c = self.game.get_random()
			if self.game.feasible(c): break
		return c

	#mode = 2: select most probable
	#----------------------------------------------------------------------
	def pure_probability(self):
		self.game.getCurrentFS()
		### choose lexicographically ###

		topIDX = [np.argsort(self.game.currentFS[1])][0]
		topC = self.game.currentFS[0][topIDX][::-1]
		return topC[0]

	#mode = 3: sharma mittal
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
		if cs.shape[0] == 1 and cs.ndim == 2:
			return cs[0]
		else:
			return cs
		
	#mode = 4: mixed strategy
	#----------------------------------------------------------------------
	def mixed_strategy(self):
		self.game.getCurrentFS()
		cs = self.game.best_mixes(t=self.t, r=self.r, p=self.p)
		print 

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


########################
# # CODE FOR TESTING
# ######################
# game = Game(codelength=3, codejar=[8,6,4,2], logging=True)
# game.initialize()
# game.guess(combination = [1,3,2])

# agent = AppAgent(game=game, mode=4)
# agent.play()

# guess = agent.compute_guess()
# game.guess(combination=guess)
# game.compute_console_statistics()

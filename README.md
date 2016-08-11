# Mastermind 
## Part of 'Entropy-based Approaches to Mastermind'

### Short project description 

We consider a variation of the well-known game of Mastermind to explore entropy-based approaches to the design of effective playing strategies and to understand human intuitions about informative queries. In the original game, the code maker composes a code of fixed length using a combination of different-colored pegs (or marbles). The code breaker takes a guess at the code, interprets the feedback provided by the code maker, and uses this information to devise her next guess. The code breaker’s goal is to guess the code in as few rounds as possible. A generic playing strategy consists of procedures for (i) identifying the set of feasible combinations, where prior feedback is used to determine which combinations are still viable and which not; and (ii) picking the combination that best serves the goal of reducing the number of feasible solutions as much as possible. 

In our proposed modification, the code maker generates the code by sampling (with replacement) from a code jar that contains an arbitrary number of pegs of different colors. Each color is chosen with probability proportional to the number of pegs of that color in the code jar. This entails that the probabilities of different candidate solutions in the feasible set might differ. Whereas in the regular version each combination is equally likely, in the modified version, a non-uniform distribution over colors in the code jar will lead to some combinations being more probable than others. To illustrate how this might impact the design of effective strategies, consider how an algorithm might choose between two alternative guesses, each resulting in a set of five feasible combinations, but where the probabilities of codes in set one and set two are [0.6, 0.1, 0.1, 0.1, 0.1] and [0.47, 0.47, 0.02, 0.02, 0.02], respectively. Which set would you rather play with going forward?

In order to quantify each set's uncertainty with respect to the hidden code, we use different entropy measures from a unified mathematical formalism called Sharma-Mittal generalized entropy. Our ultimate goal is to translate our findings to the domain of human psychology in order to investigate human intuitions about which queries are informative when playing Mastermind.


### Requirements

The application is coded in python and requires the following modules to run:
* wxPython (for GUI; https://wiki.wxpython.org/How%20to%20install%20wxPython)
* numpy (http://docs.scipy.org/doc/numpy/user/install.html)
* [optional: matplotlib and seaborn]

### Overview

**mmind-app.py** is a GUI-based application to play the Mastermind game in 
the code jar variation (composition of code jar from which code will be 
selected at random can be determined by the player, i.e., number of 
different colors and relative frequency). [See screenshot]

![screenshot][logo]

[logo]: img/ss.png "game screenshot"

Besides playing the game (as a human player) it is also possible to watch 
the computer play the game using different strategies. Among them are:
* random feasible (generate codes at random and use the first code that
is feasible)
* pure probability (pick the code the has the highest probability of
being the hidden code)
* pure information gain (pick the code the has the highest expected information
* gain under an entropy function instantiated through the specified oder-degree pair)
* mixed strategy (use mix parameter between 0 and 1 to determine influence of 
probability component and information gain component in strategy selection)

Functionality of the Mastermind App:
* 'New' (start a new game after specifying game setup, i.e., code length, 
code jar, human vs. computer player, etc.)
* 'Stats' (open statistics widget; attention: the statistics will only be 
computed (and updated) when the window is opened / as long as it's open.
Opening the window can therefore sometimes lag when opening for the first 
time)
* 'End' (click to end game)

Drag-and-drop marbles from peg box to desired location (from bottom to top; 
first row at the bottom); feedback to the right; game automatically ends 
upon correctly guessing the code


## TODOs

* (mastermind.py) strategies: choose random vs. choose lexicographic from equivalent combinations
* (mmind_app.py) integrate data saving functionalty
* make game available online and collect data from human players (see: http://stackoverflow.com/questions/16676962/wxpython-gui-to-a-website for development of wxPython-based web-app)
* run simulation studies comparing various playing strategies under different 
code jars (see poster.pdf)

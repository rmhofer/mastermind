# Mastermind Project 

### Requirements

The application requires the following python modules to run:
* wxPython (for GUI)
* numpy
* (optional: matplotlib and seaborn)

### Overview

**mmind-app.py** is a GUI-based application to play the Mastermind game in 
the code jar variation (composition of code jar from which code will be 
selected at random can be determined by the player, i.e., number of 
different colors and relative frequency). [See screenshot]

![screenshot][logo]

[logo]: img/ss.png "game screenshot"

Besides playing the game oneself (human plater) it is also posible to watch 
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

* strategies: choose random vs. choose lexicographic from equivalent combinations
* integrate data saving functionalty
* make game available online and collect data from human players
* run simulation studies comparing various playing strategies under different 
code jars

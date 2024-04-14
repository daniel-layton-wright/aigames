# AIGames

For questions, suggestions, to collaborate, etc. contact me at <dlwright@alumni.stanford.edu> .

The `aigames` package contains implementations of AI algorithms to
play games, but is primarily focused on AlphaZero.

For a fun demo of a neural network playing the game **2048**, visit:
https://daniel-layton-wright.github.io/2048ai!

AlphaZero is a reinforcement learning algorithm that learns to play games entirely through self-play.
That is, it does not depend on any domain-specific heuristics. It essentially only knows the rules of the game,
meaning how the state of the game changes when a given move is taken and what the rewards are. It needs this 
information so that it can do its tree search, which roughly corresponds to hypothetically looking into the future
before making a move, much like a human would.

There are two main pieces ot the AlphaZero algorithm.

1. Monte Carlo tree search, or MCTS: MCTS is a tree search algorithm that
   relies on a function that can output a suggested policy, **$\pi(s)$** as a function
   of a state, **$s$**, and a value $v(s)$. The value is meant to represent the expected
   future rewards from the state. With these two functions, MCTS explores the tree in a way
   that prioritizes states given a high probability by the policy or states with a high discovered value
   and, usually, some random exploration.
2. A neural network that is trained to predict the policy and value. Based on the history of 
games the algorithm has played against itself, it has data from the tree search for the policy. The network is trained 
to predict the output of the MCTS search when given the input state s. Similarly, the history of rewards is kept
and the network is trained to predict the future discounted rewards from each state s.

## AlphaZero with 2048

https://play2048.co/

The game **2048** is a single player game played in a 4x4 grid, which contains numerical tiles.
The player can move all the tiles on the board left, right, up, or down. When two neighboring tiles
have the same value and are moved in their neighboring direction, they combine into a single tile
with the sum of the previous values. So two 2s become a 4, two 4s become an 8, and so on. The player's score increases by the value of the new tile. Whenever a move is made a new tile
appears on the board in a uniform random open slot and the new tile has a 90% chance of being a 2 and a 10%
chance of being a 4. The board starts with two random tiles.

This game differs from games like Go, Chess, Connect4, TicTacToe in ways that present a number of challenges to the original implementation of AlphaZero:

* It is a single player game
* It is not simply a win/lose game (like Go, Chess, Connect4, TicTacToe), but has a score which can range into the 100,000s.
* It is much longer than those other games. A game can be many 1000s of moves.
* There is randomness: the location of the new tile at each move is random.

Because of this, the standard AlphaZero algorithm will not work on 2048 (I tried). There are a number of modifications that have to be made:
* The tree search has to account for the randomness. This is done with what are called chance nodes; I also 
call them environment nodes. These nodes represent the state of the game before the ''chance'' event has happened. In the case of 2048
they are the state of the game before the new tile appears. The children of this node are the states
of the game that can occur when one of the random locations for the new tile is chosen.
* Traditional AlphaZero's value target is usually chosen to be 1 if the game ended in a victory, -1 for a loss, 0 for a draw,
or some similar scheme. A naive first thought on modifiying this for 2048 would be to use the 
observed discounted reward from the 2048 game. In practice it is better to use a $TD(\lambda)$ estimate. The TD estimate
is a combination of the observed rewards and the predicted values of the intermediate states that occured during the game.
This stabilizes the value targets.
* The formula used in MCTS is very important
* Scale the values in the MCTS tree search: AlphaZero was developed for games where the outcome is a simple win/lose (or draw) and is typically represented by
a value like 0/1 or -1/0/1.
2048's score takes on a much greater range of values. Because of this, the tree search has to be modified to scale
the values in the tree to range between 0 and 1 based on the minimum and maximum values seen in the tree so far.
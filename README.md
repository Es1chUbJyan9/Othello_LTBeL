# Othello_LTBeL
## Features
- Tree parallelization Monte-Carlo tree search
- Lock-free multi-thread with virtual loss
- MCTS-Minimax hybrids
- Residual neural network evaluation based on Reinforcement learning
- Large-batch inference
- Using TensorFlow C++ API
- Efficient implementation in C++

## Award
- [International Computer Games Association 2018 Computer Olympiad](https://www.tcga.tw/icga-computer-olympiad-2018/en/) - Othello10x10 Bronze
- [Taiwanese Association for Artificial Intelligence 2018 Computer Game Tournaments](https://www.tcga.tw/taai2018/en/) - Othello10x10 Gold

## Requirements:
- bazel 0.15+
- tensorflow 1.0+

# Installation
1. Download the Bazel binary installer named bazel--installer-linux-x86_64.sh from [GitHub](https://github.com/bazelbuild/bazel/releases).
2. Run the Bazel installer as follows:
```
chmod +x bazel-<version>-installer-linux-x86_64.sh
./bazel-<version>-installer-linux-x86_64.sh --user
```
3. Clone the TensorFlow source code
```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
./configure
```
4. Clone the Othello_LTBeL source code
```
git clone https://github.com/Es1chUbJyan9/Othello_LTBeL.git
mv -r Othello_LTBeL/ tensorflow/
```

# Usage
- Play game
```
bash Run_Game.sh
```
- Create training data (about 5000 min)
```
bash Create_History.sh
```


## References
- Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., ... & Chen, Y. (2017). Mastering the game of Go without human knowledge. Nature, 550(7676), 354.
- Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & Lillicrap, T. (2017). Mastering chess and shogi by self-play with a general reinforcement learning algorithm. arXiv preprint arXiv:1712.01815.
- Liskowski, P., Jaskowski, W. M., & Krawiec, K. (2018). Learning to play Othello with deep neural networks. IEEE Transactions on Games.
- Chaslot, G. M. B., Winands, M. H., & van Den Herik, H. J. (2008, September). Parallel monte-carlo tree search. In International Conference on Computers and Games (pp. 60-71). Springer, Berlin, Heidelberg.
- Hingston, P., & Masek, M. (2007, September). Experiments with Monte Carlo Othello. In Evolutionary Computation, 2007. CEC 2007. IEEE Congress on (pp. 4059-4064). IEEE.
- Baier, H., & Winands, M. H. (2015). MCTS-minimax hybrids. IEEE Transactions on Computational Intelligence and AI in Games, 7(2), 167-179.
- Liu, Y. C., & Tsuruoka, Y. (2016). Asymmetric Move Selection Strategies in Monte-Carlo Tree Search: Minimizing the Simple Regret at Max Nodes. arXiv preprint arXiv:1605.02321.
- Rosenbloom, P. S. (1982). A world-championship-level Othello program. Artificial Intelligence, 19(3), 279-320.
- Buro, M. (1997). An evaluation function for othello based on statistics. Technical Report 31, NEC Research Institute.
- Buro, M. (1995, March). Logistello: A strong learning othello program. In 19th Annual Conference Gesellschaft für Klassifikation eV (Vol. 2).
- Liskowski, P., Jaskowski, W. M., & Krawiec, K. (2018). Learning to play Othello with deep neural networks. IEEE Transactions on Games.

## License
[GNU General Public License v3.0](https://github.com/Es1chUbJyan9/Othello_LTBeL/blob/master/LICENSE)

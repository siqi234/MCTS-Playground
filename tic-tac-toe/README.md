# My notes


## 📝About this notes
This is some notes I toke when I write and explore the code. I will also add questions/thoughts/problems I met in this notes. 


## 📜Agenda
- [Mid-game situation](#mid-game-situation)


## Problems
In this section, I will put different problems and explanation of my code as well.



## Mid-game situation
> Goal: choose the best next action given a mid-game board state.

### Breakdown of the codes

**Import libraries**
```
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
```

**Define Tic-Tac-Toe Game Stats**

We will have:

- A board of 3x3 grid, 9 cells
    - Stored as a length-9 list/tuple of integers:
        - `0` = empty
        - `+1` = X
        - `-1` = O
    - Indexing is row-major (top-left to bottom-right): positions `0..8`

Example: empty board

            0 0 0   |   . . .
            0 0 0   |   . . .
            0 0 0   |   . . .

- Player to move
    - `player = +1` means **X** to move
    - `player = -1` means **O** to move

Example: one move by each player (assume user is always X)

            +1  0  0   |   X . .
             0  0  0   |   . . .
             0 -1  0   |   . O .

# My notes


## 📝About this notes
This is some notes I toke when I write and explore the code. I will also add questions/thoughts/problems I met in this notes. 


## 📜Agenda
- [Mid-game situation](#mid-game-situation)


# Problems
In this section, I will put different problems and explanation of my code as well.



# Mid-game situation
> Goal: choose the best next action given a mid-game board state.

## Breakdown of the codes

**Import libraries**
```
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
```

### Define Tic-Tac-Toe Game Stats

We will have:

    @dataclass(frozen=True)
    class TTTState:
    
        board: Tuple[int, ...] = field(default_factory=lambda: (0,)*9)
        player: int = +1            # +1 or -1

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

**WIN_LINES**
    
    WIN_LINES = [
            (0,1,2),(3,4,5),(6,7,8), # horizontals 
            (0,3,6),(1,4,7),(2,5,8), # verticals 
            (0,4,8),(2,4,6)          # diagonals 
        ]
- Save the lines that once occupied player wins. 

**def legal_action(state) -> List[int]**
    
    return [i for i, v in enumerate(s.board) if v==0 ]

- Returns all valid moves from the current state.
- A move is legal if the target cell is empty/0. The function scans the 9 cells and returns the indices of empty ones.

**def next_state(s: state, a: action) -> s2: state**

    b = list(s.board)
    b[a] = s.player

    return TTTState(board=tuple(b), player=-s.player)

- Return the next state $s_2$ (the board and the player) after taking the action $a$ at the current state $s$.

**def winner(board: Tuple[int, ...]) -> int**

    for (i, j, k) in WIN_LINES:
        s = board[i] + board[j] + board[k]
        if s == 3:
            return +1
        if s == -3:
            return -1
    return 0

- Return +1 if X wins, -1 if O wins, and 0 if no winner

**def is_terminal(s: TTTState) -> bool**

    return winner(s.board) !=0 or all(v !=0 for v in s.board)

-  Return $True$ if there is a winner, otherwise continue searching

**def terminal_value_from_X(s: TTTState) -> float**

    w = winner(s.board)
    if w == +1:
        return 1.0
    if w == -1:
        return -1.0
    return 0.0

- Returns a number (float) that represents the game outcome from X’s perspective.
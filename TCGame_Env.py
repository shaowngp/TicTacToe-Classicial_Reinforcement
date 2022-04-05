### Tic-Tac-Toe Agent

from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

    def __init__(self):
        """initialising the board"""
        
        # Initializing board position
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # Initializing all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)]

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum i.e. 15
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""

       
        if (sum(curr_state[0:3:1])==15 or sum(curr_state[3:6:1])==15 or 
            sum(curr_state[6:9:1])==15 or
            sum(curr_state[0:9:3])==15 or 
            sum(curr_state[1:9:3])==15 or 
            sum(curr_state[2:9:3])==15 or
            sum(curr_state[0:9:4])==15 or 
            sum(curr_state[2:8:2])==15):
           
            return True
        
        else:
            return False
 

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
       
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        next_state = [i for i in curr_state]
        next_state[curr_action[0]] = curr_action[1]

        return next_state


    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        
        next_state = self.state_transition(curr_state, curr_action)
        
       
        in_terminal_state, message = self.is_terminal(next_state)
        
        if in_terminal_state:
            
            if message == "Win":
                reward = 10
                episode_msg = "Agent has won!"
            else:
                reward = 0
                episode_msg = "Tie!"
                
            return (next_state, reward, in_terminal_state, episode_msg)
        else:
         
            
            _, env_actions = self.action_space(next_state)
            
            env_action = random.choice([ac for i, ac in enumerate(env_actions)])
            
            next_new_state = self.state_transition(next_state, env_action)
            
            in_terminal_state, message = self.is_terminal(next_new_state)
            
            if in_terminal_state:
                if message == "Win":
                    reward = -10
                    episode_msg = "Environment has won!"
                else:
                    reward = 0
                    episode_msg = "Tie!"
                    
            else:
                reward = -1
                episode_msg = "Resume"
         
            return (next_new_state, reward, in_terminal_state, episode_msg)

    def reset(self):
        return self.state

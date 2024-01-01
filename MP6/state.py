import copy
import math
import numpy as np
from itertools import count

# NOTE: using this global index means that if we solve multiple
#       searches consecutively the index doesn't reset to 0... this is fine
global_index = count()


# TODO VI
# Euclidean distance between two state tuples, of the form (x,y, shape)
def euclidean_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a[:2] - b[:2])


from abc import ABC, abstractmethod


class AbstractState(ABC):
    def __init__(self, state, goal, dist_from_start=0., use_heuristic=True):
        self.state = state
        self.goal = goal
        # we tiebreak based on the order that the state was created/found
        self.tiebreak_idx = next(global_index)
        # dist_from_start is classically called "g" when describing A*, i.e., f = g + h
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.h = self.compute_heuristic()
        else:
            self.h = 0

    # To search a space we will iteratively call self.get_neighbors()
    # Return a list of State objects
    @abstractmethod
    def get_neighbors(self):
        pass

    # Return True if the state is the goal
    @abstractmethod
    def is_goal(self):
        pass

    # A* requires we compute a heuristic from eahc state
    # compute_heuristic should depend on self.state and self.goal
    # Return a float
    @abstractmethod
    def compute_heuristic(self):
        pass

    # The "less than" method ensures that states are comparable
    #   meaning we can place them in a priority queue
    # You should compare states based on f = g + h = self.dist_from_start + self.h
    # Return True if self is less than other
    @abstractmethod
    def __lt__(self, other):
        # NOTE: if the two states (self and other) have the same f value, tiebreak using tiebreak_idx as below
        if self.tiebreak_idx < other.tiebreak_idx:
            return True

    # __hash__ method allow us to keep track of which
    #   states have been visited before in a dictionary
    # You should hash states based on self.state (and sometimes self.goal, if it can change)
    # Return a float
    @abstractmethod
    def __hash__(self):
        pass

    # __eq__ gets called during hashing collisions, without it Python checks object equality
    @abstractmethod
    def __eq__(self, other):
        pass

# NOTE: unlike MP 3/4, in this MP, the maze might not have a path! So be sure to handle this case properly and return None!

# To complete the MazeState code, you will need to complete:

# euclidean_distance(a,b): which is similar to manhattan distance from MP 4
# get_neighbors(self): returns the MazeState neighbors using self.maze_neighbors. When computing move (edge) costs, note that changing shape has a cost of 10.
# __hash__(self): hash the MazeState using the centroid, shape, and goals.
# __eq__(self): check if two states are equal
# compute_heuristic(self): compute the heuristic, which is the euclidean distance to the nearest goal.
# __lt__(self, other): This method allows the heap to sort States according to f = g + h value.

# State: a length 3 list indicating the current location in the grid and the shape
# Goal: a tuple of locations in the grid that have not yet been reached
#   NOTE: it is more efficient to store this as a binary string...
# maze: a maze object (deals with checking collision with walls...)
class MazeState(AbstractState):
    def __init__(self, state, goal, dist_from_start, maze, use_heuristic=True):
        # NOTE: it is technically more efficient to store both the mst_cache and the maze_neighbors functions globally,
        #       or in the search function, but this is ultimately not very inefficient memory-wise
        self.maze = maze
        self.maze_neighbors = maze.get_neighbors
        super().__init__(state, goal, dist_from_start, use_heuristic)

    # TODO VI
    def get_neighbors(self):
        # if the shape changes, it will have a const cost of 10.
        # otherwise, the move cost will be the euclidean distance between the start and the end positions
        nbr_states = []
        
        neighboring_locs = self.maze_neighbors(*self.state)
        print("neighbor", neighboring_locs)
        print("goal", self.goal)
                
        cost_tot = 10
        # Create GridState objects for all neighbors
        for neighbor in neighboring_locs: 
            goal_list_updated = self.goal
            # If neighbor is a goal state, we want to remove it from set of goals to still find
            if (neighbor[0], neighbor[1]) in self.goal: 
                goal_list_updated = ()
                # for goal in self.goal:
                #     if goal != neighbor:
                #         goal_list_updated.append(goal)
                # goal_list_updated = tuple(goal_list_updated)
            if neighbor[2] == self.state[2]: 
                cost_tot = euclidean_distance((self.state[0], self.state[1]), (neighbor[0], neighbor[1]))
            # (self, state, goal, dist_from_start, maze, use_heuristic=True)
            nbr_states.append(MazeState(self.state, goal_list_updated, self.dist_from_start + cost_tot, self.maze, self.use_heuristic))
        return nbr_states

    # TODO VI
    def is_goal(self):
        # if self.state not in self.goal: 
        #     return False
        # return True
        return len(self.goal) == 0

    # We hash BOTH the state and the remaining goals
    #   This is because (x, y, h, (goal A, goal B)) is different from (x, y, h, (goal A))
    #   In the latter we've already visited goal B, changing the nature of the remaining search
    # NOTE: the order of the goals in self.goal matters, needs to remain consistent
    # TODO VI
    def __hash__(self):
        return hash((self.state[0], self.state[1], self.state[2], tuple(self.goal)))

    # TODO VI
    def __eq__(self, other):
        return self.state == other.state

    # Our heuristic is: distance(self.state, nearest_goal)
    # We euclidean distance
    # TODO VI
    def compute_heuristic(self):
        min_val = float('inf')
        for goal in self.goal: 
            dist = euclidean_distance((self.state[0], self.state[1]), (goal[0], goal[1]))
            if min_val > dist: 
                min_val = dist
        
        return min_val

    # This method allows the heap to sort States according to f = g + h value
    # TODO VI
    def __lt__(self, other):
        print("got here yo")
        # same as for the Wordladder
        curr_cost = self.dist_from_start + self.compute_heuristic()
        other_cost = other.dist_from_start + other.compute_heuristic()
        print(curr_cost)
        print(other_cost)
        if curr_cost < other_cost:   
            return True
        # specific condition to check for tie in cost, did in separate line for readability.
        elif curr_cost == other_cost and self.tiebreak_idx < other.tiebreak_idx: 
            return True
        
        return False

    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goal)

    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goal)

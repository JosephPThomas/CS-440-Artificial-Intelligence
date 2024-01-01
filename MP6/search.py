# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq


# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI
def astar(maze):
    starting_state = maze.get_start()
    
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search algorithm
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state) and obtain the path
    #   - keep track of the distance of each state from start node
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    # Your code here ---------------------------------------------

    # 1. Retrieve the starting state's neighbors
    # 2. Enqueue the starting state's neighbors
    # 3. Push all new neighbors onto hashmap
    # 4. Check whether the front of the queue is is_goal(), if so, call backtrack and skeddadle
    # 5. Go again from step 1 with item at front of queue

    current_state = starting_state # declare current state var

    # we want to keep going with our search until we have no more frontiers or our least cost node is the goal node
    while len(frontier) > 0: 
        # analyze the lowest cost neighbor
        shortest_node = heapq.heappop(frontier)
        if shortest_node.is_goal(): # if this is the case, then we're done with the search
            print("why ammi here")
            return backtrack(visited_states, shortest_node)
        current_state = shortest_node # the lowest cost node is now the node whose neighbors we'll search 
        # retrieve the neighbors to go through 
        neighbors = current_state.get_neighbors()
        # add the neighbors to our table and priority queue
        for node in neighbors:
            # cost for A*
            node_cost = node.dist_from_start + node.h 
            # check whether our current node is new or we have a better cost for it, then do something
            if node not in visited_states or visited_states[node][1] > node_cost:
                visited_states[node] = (current_state, node_cost)
                heapq.heappush(frontier, node)
            else: 
                # if we have found an already found node and we don't even have a good cost for it, just skip to next neighbor
                continue
    
    # if you do not find the goal return an empty list
    return []


# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
# TODO: VI
def backtrack(visited_states, current_state):
    path = []
    # print(visited_states)
    # Your code here ---------------------------------------------

    # 1. Retrieve the end goal's parent
    # 2. enqueue parent onto path
    # 3. repeat until parent is the start state (dist from start = 0)
    # 4. flip all elements of list 

    # just backtrack the hash map. Each key has a value of its parent so we build the list
    state = current_state 
    while visited_states[state][1] != 0: 
        path.append(state)
        state = visited_states[state][0]

    path.append(state) # append the start state too

    return path[::-1] # nifty trick to flip the list from start to goal 

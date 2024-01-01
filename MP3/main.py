from state import WordLadderState, EightPuzzleState
from search import best_first_search
from utils import read_puzzle, read_word_ladders

import time
import argparse

def main(args):
    if args.problem_type == "WordLadder":
        word_ladder_problems = read_word_ladders()
    
        for start_word, goal_word in word_ladder_problems:
            print(f"Doing WordLadder from {start_word} to {goal_word}")
            start = time.time()
            starting_state = WordLadderState(start_word, goal_word, dist_from_start=0, use_heuristic=args.use_heuristic)
            path = best_first_search(starting_state)
            end = time.time()
            print("\tPath length: ", len(path))
            # print("\tUnique states visited: ", len(visited))
            print("\tPath found:", [p for p in path])
            print(f"\tTime: {end-start:.3f}")

    elif args.problem_type == "EightPuzzle":
        print(f"Doing EightPuzzle for length {args.puzzle_len} puzzles")
        all_puzzles = read_puzzle(f"data/eight_puzzle/{args.puzzle_len}_moves.txt")
        for puzzle in all_puzzles:
            start = time.time()
            start_puzzle = puzzle[0]
            zero_loc = puzzle[1]
            print(f"Start puzzle: {start_puzzle}")
            # import numpy as np
            # goal_puzzle = np.arange(9).reshape(3,3).tolist()
            goal_puzzle = [[0,1,2],[3,4,5],[6,7,8]]
            starting_state = EightPuzzleState(start_puzzle, goal_puzzle, 
                                dist_from_start=0, use_heuristic=args.use_heuristic, zero_loc=zero_loc)
            path = best_first_search(starting_state)
            end = time.time()
            print("\tPath length: ", len(path))
            # print("\tUnique states visited: ", len(visited))
            # print("\tPath found:", [p for p in path])
            print(f"\tTime: {end-start:.3f}")
    else:
        print("Problem type must be one of [WordLadder, EightPuzzle]")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP3 Search')
    # WORDLADDER ARGS
    parser.add_argument('--problem_type',dest="problem_type", type=str,default="WordLadder",
                        help='Which search problem (i.e., State) to solve: [WordLadder, EightPuzzle]')
    
    # EIGHTPUZZLE ARGS
    parser.add_argument('--puzzle_len',dest="puzzle_len", type=int, default = 5,
                        help='EightPuzzle problem difficulty: one of [5, 10, 27]')
    
    parser.add_argument('--use_heuristic', default = True, action = 'store_true',
                        help = 'use heuristic h in best_first_search')
    
    args = parser.parse_args()
    main(args)
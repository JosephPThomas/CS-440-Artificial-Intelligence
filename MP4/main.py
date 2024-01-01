from state import SingleGoalGridState, GridState
from search import best_first_search
from maze import Application, Maze

import time
import argparse

def main(args):

    if "Grid" in args.problem_type:
        # MAZE -------
        filename = args.maze_file
        print(f"Doing Maze search for file {filename}")
        maze = Maze(filename)
        path = []
        if not args.human:
            start = time.time()  
            if args.problem_type == "GridSingle":
                starting_state = SingleGoalGridState(maze.start, maze.waypoints, 
                                dist_from_start=0, use_heuristic=args.use_heuristic,
                                maze_neighbors=maze.neighbors)
            else:
                starting_state = GridState(maze.start, maze.waypoints, 
                                dist_from_start=0, use_heuristic=args.use_heuristic,
                                maze_neighbors=maze.neighbors, mst_cache={})
            path = best_first_search(starting_state)
            end = time.time()

            print("\tGoals: ", maze.waypoints)
            print("\tStart: ", maze.start)
            print("\tPath length: ", len(path))
            # print("\tUnique states visited: ", len(visited))
            print("\tStates explored: ", maze.states_explored)
            # print("Path found:", [p for p in path])
            print("\tTime:", end-start)
        
        if args.show_maze_vis or args.human:
            path = [s.state for s in path]
            application = Application(args.human, args.scale, args.fps, args.altcolor)
            application.run(maze, path, args.save_maze)
    else:
        #print("Problem type must be one of [WordLadder, EightPuzzle, GridSingle, GridMulti]")
        print("Problem type must be one of [GridSingle, GridMulti]")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP4 Search')
    # WORDLADDER ARGS
    parser.add_argument('--problem_type',dest="problem_type", type=str,default="GridSingle",
                        help='Which search problem (i.e., State) to solve: [GridSingle, GridMulti]')
    
    # MAZE ARGS
    parser.add_argument('--maze_file', type=str, default="data/mazes/grid_single/tiny",
                        help = 'path to maze file')
    parser.add_argument('--show_maze_vis', default = False, action = 'store_true',
                        help = 'show maze visualization')
    parser.add_argument('--human', default = False, action = 'store_true',
                        help = 'run in human-playable mode')
    parser.add_argument('--use_heuristic', default = True, action = 'store_true',
                        help = 'use heuristic h in best_first_search')
    
    # You do not need to change these
    parser.add_argument('--scale',  dest = 'scale', type = int, default = 20,
                        help = 'display scale')
    parser.add_argument('--fps',    dest = 'fps', type = int, default = 30,
                        help = 'display framerate')
    parser.add_argument('--save_maze', dest = 'save_maze', type = str, default = None,
                        help = 'save output to image file')
    parser.add_argument('--altcolor', dest = 'altcolor', default = False, action = 'store_true',
                        help = 'view in an alternate color scheme')

    args = parser.parse_args()
    main(args)
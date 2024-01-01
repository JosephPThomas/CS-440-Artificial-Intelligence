import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table. 
        self.N[state + (action,)] += 1
        return

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table. 
        prime_state = tuple(s)
        state = tuple(s_prime)

        N = self.N[prime_state + (a,)]
        learning_rate = self.C / (self.C + N)

        old_value = self.Q[prime_state + (a,)]
        maxQ = np.max([self.Q[state + (action,)] for action in self.actions])

        Q_update = old_value + learning_rate * (r + self.gamma * maxQ - old_value)

        self.Q[prime_state + (a,)] = Q_update
        return       

    def collide(self, previous_state):
        adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right = \
        previous_state[2], previous_state[3], previous_state[4], previous_state[5], previous_state[6], previous_state[7]

        if (adjoin_wall_x == 1 and self.a == 2) or (adjoin_wall_x == 2 and self.a == 3) or \
        (adjoin_wall_y == 1 and self.a == 0) or (adjoin_wall_y == 2 and self.a == 1):
            return True

        elif (self.a == 0 and adjoin_body_top) or (self.a == 1 and adjoin_body_bottom) or \
            (self.a == 2 and adjoin_body_left) or (self.a == 3 and adjoin_body_right):
            return True

        else:
            return False
        
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        # TODO - MP12: write your function here

        reward = self.calculate_reward(points, dead)
        s_prime = self.generate_state(environment)

        if self._train:
            if self.s is not None and self.a is not None:
                self.update_n(self.s, self.a)
                self.update_q(self.s, self.a, reward, s_prime)

            if dead:
                self.reset()
                action = np.random.choice(self.actions)
            else:
                opt_action = utils.RIGHT
                food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right = s_prime
                opt_f = self.Q[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, opt_action]
                for action in [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP]:
                    if self._train:
                        visit_times = self.N[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, action]
                        if visit_times < self.Ne:
                            f = 1 
                        else:
                            f = self.Q[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, action]
                    else:
                        f = self.Q[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, action]
                    if f > opt_f:
                        opt_action = action
                        opt_f = f
                action = opt_action
                self.s = s_prime
                self.a = action
                self.points = points
        else:
            opt_action = utils.RIGHT
            food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right = s_prime
            opt_f = self.Q[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, opt_action]
            for action in [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP]:
                f = self.Q[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, action]
                if f > opt_f:
                    opt_action = action
                    opt_f = f
            action = opt_action

        return action

    def calculate_reward(self, points, dead):
        if points == self.points + 1:
            return 1
        elif dead:
            return -1
        else:
            return -0.1

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # TODO - MP11: Implement this helper function that generates a state given an environment 

        snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y = environment

        food_dir_x = 1 if food_x < snake_head_x else (2 if food_x > snake_head_x else 0)
        food_dir_y = 1 if food_y < snake_head_y else (2 if food_y > snake_head_y else 0)

        adjoining_wall_x = 1 if snake_head_x == 1 or (snake_head_x == rock_x + 2 and snake_head_y == rock_y) else (2 if snake_head_x == self.display_width - 2 or (snake_head_x == rock_x - 1 and snake_head_y == rock_y) else 0)

        adjoining_wall_y = 1 if snake_head_y == 1 or (snake_head_x == rock_x and snake_head_y == rock_y + 1) else (2 if snake_head_y == self.display_height - 2 or (snake_head_x == rock_x and snake_head_y == rock_y - 1) else 0)

        adjoining_body_top = 1 if (snake_head_x, snake_head_y - 1) in snake_body else 0
        adjoining_body_bottom = 1 if (snake_head_x, snake_head_y + 1) in snake_body else 0
        adjoining_body_left = 1 if (snake_head_x - 1, snake_head_y) in snake_body else 0
        adjoining_body_right = 1 if (snake_head_x + 1, snake_head_y) in snake_body else 0

        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)

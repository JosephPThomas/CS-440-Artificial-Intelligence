# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
import math
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    if alien.is_circle():
        return circle_touches_a_wall(alien, walls)
    else:
        return alien_touches_a_wall(alien, walls)

def circle_touches_a_wall(alien: Alien, walls: List[Tuple[int]]) -> bool:
    circle_centroid = alien.get_centroid()
    circle_width = alien.get_width()
    
    for wall in walls:
        wall_segment = get_wall_segment(wall)
        if point_segment_distance(circle_centroid, wall_segment) <= circle_width:
            return True

    return False

def alien_touches_a_wall(alien: Alien, walls: List[Tuple[int]]) -> bool:
    alien_segment = alien.get_head_and_tail()
    alien_width = alien.get_width()
    
    for wall in walls:
        wall_segment = get_wall_segment(wall)
        
        if segment_distance(wall_segment, alien_segment) <= alien_width:
            return True

    return False

def get_wall_segment(wall: Tuple[int]) -> Tuple[Tuple[int]]:
    start_point  = (wall[0], wall[1])
    end_point  = (wall[2], wall[3])
    return (start_point , end_point )

def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    collision_points = [(0, 0, 0, window[ 1 ]), (0, window[ 1 ], window[ 0 ], window[ 1 ]), 
                (window[ 0 ], window[ 1 ], window[ 0 ], 0), (window[ 0 ], 0, 0, 0)]
    

    if does_alien_touch_wall(alien, collision_points):
        return False

    if alien.is_circle():
        circle_centroid = alien.get_centroid()
        return is_point_in_polygon(circle_centroid, collision_points)
    else:
        alien_segment = alien.get_head_and_tail()
        head = alien_segment[0]
        return is_point_in_polygon(head, collision_points)


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    def get_segment(i):
        start_point = polygon[i]
        end_point = polygon[(i + 1) % len(polygon)]
        return start_point, end_point

    def count_intersections(test_strip, segments):
        count_intersection = 0
        for segment in segments:
            if do_segments_intersect(test_strip, segment):
                count_intersection += 1
        return count_intersection

    segments = [get_segment(i) for i in range(len(polygon))]
    test_strip = (point, (500, point[1]))
    count_intersection = count_intersections(test_strip, segments)

    return count_intersection % 2 == 1

def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    if does_alien_touch_wall(alien, walls):
        return True

    start = alien.get_centroid()
    end = waypoint

    if start == end:
        return False

    alien_width = alien.get_width()

    for wall in walls:
        movement = (start, end)
        wall_intersection = ((wall[0], wall[1]), (wall[2], wall[3]))

        if segment_distance(movement, wall_intersection) <= alien_width:
            return True

    if alien.is_circle():
        return False

    movement_delta  = (end[0] - start[0], end[1] - start[1])

    alien_segment = alien.get_head_and_tail()
    alien_start_beginning, alien_start_ending = alien_segment
    alien_end_beginning = (alien_start_beginning[0] + movement_delta [0], alien_start_beginning[1] + movement_delta [1])
    alien_end_ending = (alien_start_ending[0] + movement_delta [0], alien_start_ending[1] + movement_delta [1])
    collision_points = (alien_start_beginning, alien_start_ending, alien_end_ending, alien_end_beginning)

    if movement_delta [0] == 0 and alien_start_beginning[0] == alien_start_ending[0]:
        return movement_pointer(alien, walls, movement_delta ) 
    if movement_delta [1] == 0 and alien_start_beginning[1] == alien_start_ending[1]:
        return movement_pointer(alien, walls, movement_delta )

    for wall in walls:
        beginning_of_wall = (wall[0], wall[1])
        ending_of_wall = (wall[2], wall[3])

        if is_point_in_polygon(beginning_of_wall, collision_points) or is_point_in_polygon(ending_of_wall, collision_points):
            return True

    for wall in walls:
        beginning_of_wall = (wall[0], wall[1])
        ending_of_wall = (wall[2], wall[3])

        if segment_distance((alien_end_beginning, alien_end_ending), (beginning_of_wall, ending_of_wall)) <= alien_width:
            return True

    for wall in walls:
        beginning_of_wall = (wall[0], wall[1])
        ending_of_wall = (wall[2], wall[3])

        for i in range(len(collision_points)):
            intersect1 = collision_points[i]

            if i == len(collision_points) - 1:
                intersect2 = collision_points[0]
            else:
                intersect2 = collision_points[i + 1]

            if do_segments_intersect((beginning_of_wall, ending_of_wall), (intersect1, intersect2)):
                return True

    return False


def movement_pointer(alien: Alien, walls: List[Tuple[int]], movement_delta: Tuple[int, int]):
    alien_start_beginning, alien_start_ending = alien.get_head_and_tail()
    alien_end_beginning = (alien_start_beginning[0] + movement_delta[0], alien_start_beginning[1] + movement_delta[1])
    alien_end_ending = (alien_start_ending[0] + movement_delta[0], alien_start_ending[1] + movement_delta[1])

    index = 1 if movement_delta[0] == 0 else 0

    if abs(alien_start_beginning[index] - alien_end_ending[index]) > abs(alien_start_ending[index] - alien_end_beginning[index]):
        line_segment = (alien_start_beginning, alien_end_ending)
    else:
        line_segment = (alien_start_ending, alien_end_beginning)

    for wall in walls:
        beginning_of_wall = (wall[0], wall[1])
        ending_of_wall = (wall[2], wall[3])

        if segment_distance(line_segment, (beginning_of_wall, ending_of_wall)) <= alien.get_width():
            return True

    return False


def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    px, py = p
    x1, y1 = s[0]
    x2, y2 = s[1]

    segment_dx, segment_dy = x2 - x1, y2 - y1

    if segment_dx == 0 and segment_dy == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    t = ((px - x1) * segment_dx + (py - y1) * segment_dy) / (segment_dx * segment_dx + segment_dy * segment_dy)

    if t < 0:
        closest_x, closest_y = x1, y1
    elif t > 1:
        closest_x, closest_y = x2, y2
    else:
        closest_x, closest_y = x1 + t * segment_dx, y1 + t * segment_dy

    dx, dy = px - closest_x, py - closest_y
    calculation = math.sqrt(dx ** 2 + dy ** 2)
    return calculation


def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    def calculate_direction(point1, point2, point3):
        cross_product = (point2[1] - point1[1]) * (point3[0] - point2[0]) - (point2[0] - point1[0]) * (point3[1] - point2[1])
        if cross_product > 0:
            return 1
        elif cross_product < 0:
            return 2
        else:
            return 0

    def segment_overlap(point1, point2, point3):
        return point2[0] <= max(point1[0], point3[0]) and point2[0] >= min(point1[0], point3[0]) and point2[1] <= max(point1[1], point3[1]) and point2[1] >= min(point1[1], point3[1])

    start_point1, end_point1, start_point2, end_point2 = s1[0], s1[1], s2[0], s2[1]

    direction1, direction2, direction3, direction4 = calculate_direction(start_point1, end_point1, start_point2), calculate_direction(start_point1, end_point1, end_point2), calculate_direction(start_point2, end_point2, start_point1), calculate_direction(start_point2, end_point2, end_point1)

    if direction1 != direction2 and direction3 != direction4:
        return True

    if direction1 == 0 and segment_overlap(start_point1, start_point2, end_point1):
        return True
    if direction2 == 0 and segment_overlap(start_point1, end_point2, end_point1):
        return True
    if direction3 == 0 and segment_overlap(start_point2, start_point1, end_point2):
        return True
    if direction4 == 0 and segment_overlap(start_point2, end_point1, end_point2):
        return True

    return False
def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect( s1, s2 ):
        return 0
    
    distance_from_s1_to_s2 = [ ]

    distance_from_s1_to_s2.append(point_segment_distance(s1[0],s2))
    distance_from_s1_to_s2.append(point_segment_distance(s1[1],s2))
    distance_from_s1_to_s2.append(point_segment_distance(s2[0],s1))
    distance_from_s1_to_s2.append(point_segment_distance(s2[1],s1))
    return min(distance_from_s1_to_s2)



if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'



    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")

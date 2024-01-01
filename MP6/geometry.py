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
# Uploading to MP5 Sheerly to debug for MP6
import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy
import math


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """ 
    
    # Figure out position of alien 
    pos = alien.get_centroid()
    # Figure out boundaries of alien
    # width will be diameter if alien is circle
    width = alien.get_width()
    length = alien.get_length()
    # ret_val = False
    for obstacle in walls: 
        # Used in all functions that determine intersection state
        endpoints = ((obstacle[0], obstacle[1]), (obstacle[2], obstacle[3]))
        # For circle, check whether distance from center to obstacle is < radius
        # print("got Here 5")
        if alien.is_circle(): 
            # print("Got Here 4")
            # print(point_segment_distance(pos, endpoints))
            if point_segment_distance(pos, endpoints) < width: 
                return True
        else: 
            # print(alien.get_head_and_tail(), endpoints)
            # print(do_segments_intersect(alien.get_head_and_tail(), endpoints))
            head, tail = alien.get_head_and_tail()
            if point_segment_distance(head, endpoints) < width: 
                return True
            if point_segment_distance(tail, endpoints) < width: 
                return True

            if alien.get_shape() == 'Horizontal': 
                if do_segments_intersect(((head[0], head[1] - width), (tail[0], tail[1] - width)), endpoints) or do_segments_intersect(((head[0], head[1] + width), (tail[0], tail[1] + width)), endpoints): 
                    return True
            
            if alien.get_shape() == 'Vertical': 
                if do_segments_intersect(((head[0] - width, head[1]), (tail[0] - width, tail[1])), endpoints) or do_segments_intersect(((head[0] + width, head[1]), (tail[0] + width, tail[1])), endpoints): 
                    return True
                          
            if do_segments_intersect(alien.get_head_and_tail(), endpoints): 
                return True

    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    # Figure out position of alien 
    pos = alien.get_centroid()
    # Figure out boundaries of alien
    # width will be diameter if alien is circle
    width = alien.get_width()
    length = alien.get_length() 
    
    if alien.is_circle(): 
        if (0 < pos[0] - width and pos[0] + width < window[0]):
            if (0 < pos[1] - width and pos[1] + width < window[1]):
                return True
    else: 
        # Check whether endpoints are in window
        endpoints = alien.get_head_and_tail()
        if (0 < endpoints[0][0] - width and endpoints[0][0] + width < window[0]):
            if (0 < endpoints[0][1] - width and endpoints[0][1] + width < window[1]):
                if (0 < endpoints[1][0] - width and endpoints[1][0] + width < window[0]):
                    if (0 < endpoints[1][1] - width and endpoints[1][1] + width < window[1]):
                        return True  
    return False


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    
    for pt in polygon: 
        if point == pt: 
            return True
    
    collin_cond = ccw_or_cw(polygon[0], polygon[1], polygon[2])
    
    if collin_cond == 0: 
        for i in range(len(polygon)): 
            if point_segment_distance(point, (polygon[i], polygon[(i + 1) % len(polygon)])) == 0: 
                return True
        return False
    
    sum_angle = 0
    # Have to convert to numpy
    polygon = np.array(polygon)
    point = np.array(point)
    
    # Iterate through all sides
    for i in range(len(polygon)): 
        point_vect = polygon[i] - point
        # Found this clever technique when coding a queue from scratch in past
        edge = polygon[(i + 1) % len(polygon)] - point
        # Manipulate np.dot to extract calculated angle
        sum_angle += np.arccos(np.dot(point_vect, edge)/(np.linalg.norm(point_vect) * np.linalg.norm(edge)))

    # print(sum_angle)
    # Is-close function helps because floating point errors in python
    return np.isclose(sum_angle, 2 * np.pi)


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
    
    pos = alien.get_centroid()
    alien2 = deepcopy(alien)
    alien2.set_alien_pos((waypoint[0], waypoint[1]))
    head, tail = alien.get_head_and_tail()
    head2, tail2 = alien2.get_head_and_tail()
    if does_alien_touch_wall(alien2, walls): 
        return True
    for obstacle in walls:
        if do_segments_intersect(((obstacle[0], obstacle[1]), (obstacle[2], obstacle[3])), (pos, waypoint)):
            return True
        if do_segments_intersect(((obstacle[0], obstacle[1]), (obstacle[2], obstacle[3])), (head, head2)):
            return True
        if do_segments_intersect(((obstacle[0], obstacle[1]), (obstacle[2], obstacle[3])), (tail, tail2)):
            return True
    return does_alien_touch_wall(alien, walls)

def ccw_or_cw(e1, e2, e3):
    """Calculate area relation between points of segment, utilized in determining intersection

        Args:
            e1: 1st endpoint of a segment
            e2: 2nd endpoint of a segment
            e3: one endpoint of another segment

        Return:
            0, 1, -1 based on area relation of line segments
    """
    
    area = ((e2[1] - e1[1]) * (e3[0] - e2[0])) - ((e2[0] - e1[0]) * (e3[1] - e2[1]))
    # Care about the sign only 
    # threshold = 1e-3
    # if -threshold < area < threshold:
    #     return 0
    if area == 0: 
        return 0
    return np.sign(area)

def collinear_case(e1, e2, test_point):
    """Check whether a point is within the range of bounds of another segment, 
    Try for all 4 endpoints of 2 segments

        Args:
            e1: 1st endpoint of a segment
            e2: 2nd endpoint of a segment
            test_point: one endpoint of another segment

        Return:
            True if point test_point is between points e1 and e2
    """
    # Check if point is collinear with the segment
    return ((min(e1[0], e2[0]) <= test_point[0] <= max(e1[0], e2[0])) and (min(e1[1], e2[1]) <= test_point[1] <= max(e1[1], e2[1])))

def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2. Inspired by Geeks4Geeks

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    # Extract x, y coords
    s1 = np.array(s1)
    s2 = np.array(s2)
    e11 = s1[0]
    e12 = s1[1]
    e21 = s2[0]
    e22 = s2[1]
    # print(s1, s2)
    orientation1 = ccw_or_cw(e11, e12, e21)
    orientation2 = ccw_or_cw(e11, e12, e22)
    orientation3 = ccw_or_cw(e21, e22, e11)
    oritentation4 = ccw_or_cw(e21, e22, e12)
    # Figure out area relation for orientation 
    if (orientation1 != orientation2) and (orientation3 != oritentation4):
        # print("Got Here 1")
        return True

    # If collinear, check whether endpoints are between each other (parallel and intersecting)
    if ((orientation1 == 0 and collinear_case(e11, e12, e21)) or 
        (orientation2 == 0 and collinear_case(e11, e12, e22)) or
        (orientation3 == 0 and collinear_case(e21, e22, e11)) or 
        (oritentation4 == 0 and  collinear_case(e21, e22, e12))):
        # print("Got Here 2")
        return True
    # print("Got Here 3")
    return False

def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

    Args:
        p: A tuple (x, y) of the coordinates of the point.
        s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

    Return:
        Euclidean distance from the point to the line segment.
    """
    p = np.array(p)
    s = np.array(s)
    # print(s)
    line = s[1] - s[0]
    # Formula to calculate vector projection, but we use clever ratio
    projection_vector_ratio = np.dot(p - s[0], line) / np.dot(line, line)
    
    # What's conveinient about this, is that this will allow us to choose endpoints to find dist with
    projection_vector = np.clip(projection_vector_ratio, 0, 1)
    
    # Calculate the point on u
    proj_point = s[0] + projection_vector * line
    
    # find distance 
    return np.linalg.norm(p - proj_point)


def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    # If segments intersect, distance is 0
    if do_segments_intersect(s1, s2):
        return 0
    
    # Find distance from a endpoint to another point and find the closest distance 
    
    
    return min([point_segment_distance(s1[0], s2), point_segment_distance(s1[1], s2), point_segment_distance(s2[0], s1), point_segment_distance(s2[1], s1)])


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

            # Initialize Aliens and perform simple sanity check.


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

    # Test validity of straight line paths between an alien and a waypoint
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

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")

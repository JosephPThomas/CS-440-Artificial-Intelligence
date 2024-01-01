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
import pdb
from alien import Alien
from typing import List, Tuple
from copy import deepcopy

# --------------------- TO DO (MP5) --------------------------- #         
def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    # To check if the alien touches the wall, we need to check  #
    # for the current form. Handle accordingly.                 #
    if alien.is_circle( ):
        # Alien is a circle. Check if any walls fall within the #
        # radius of the circle. We can do this by checking if   #
        # distance from any of the edges is less than the       #
        # radius of the circle.                                 #

        # Get the centroid of the circle.                       #
        centroid = alien.get_centroid( )

        # For each wall, check if distance from centroid to     #
        # wall is less than width. Get width of the alien.      #
        ball_width = alien.get_width( )

        # Iterate through walls, check if distance is less than #
        # width.                                                #
        for wall in walls:
            # Set vertices of wall. We need to set as tuple for #
            # our call to point_segment_distance.               #
            vertex1 = ( wall[ 0 ], wall[ 1 ] )
            vertex2 = ( wall[ 2 ], wall[ 3 ] )
            wall_tuple = ( vertex1, vertex2 )

            # Check if distance is less than width.             #
            if point_segment_distance( centroid, wall_tuple ) <= ball_width:
                return True
    # Else, is not a circle. We need to check the oblong cases. #
    # We can use a similar approach to the circle, except we    #
    # use segment_distance to determine the distance between a  #
    # wall and the given oblong alien's line segment.           #
    else:
        # Get the head and tail of the circle. We don't care    #
        # about the order for this function.                    #
        head_and_tail = alien.get_head_and_tail( )

        # Get the width of the alien.                           #
        width = alien.get_width( )

        # Iterate through walls, check if distance is less than #
        # width. Also check for intersections ( included in the #
        # width comparison though ).                            #
        for wall in walls:
            # Set vertices of wall. We need to set as tuple for #
            # our call to point_segment_distance.               #
            vertex1 = ( wall[ 0 ], wall[ 1 ] )
            vertex2 = ( wall[ 2 ], wall[ 3 ] )
            wall_tuple = ( vertex1, vertex2 )

            # Check if distance between wall and alien's body   #
            # is less than the given width/radius.              #
            if segment_distance( wall_tuple, head_and_tail ) <= width:
                return True
    
    # Neither case true - the alien does not touch a wall.      #
    # Return False to indicate.                                 #
    return False

# --------------------- TO DO (MP5) --------------------------- #
# We will use a similar method to above, in combination with    #
# the is_point_in_polygon method. Check if the distance is      #
# close enough to the window borders. If so, determine if the   #
# alien is on the border with does_alien_touch_wall, or if the  #
# alien is in/outside the border using a combination of         #
# segment_distance and is_point_in_polyon.                      #
def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    # Initialize borders as our "walls". Format them so that    #
    # we can call does_alien_touch_wall method. Also format     #
    # them so that we can call the is_point_in_polygon method   #
    # if we have to.                                            #
    borders = [ ( 0, 0, 0, window[ 1 ] ), ( 0, window[ 1 ], window[ 0 ], window[ 1 ] ), 
                ( window[ 0 ], window[ 1 ], window[ 0 ], 0 ), ( window[ 0 ], 0, 0, 0 ) ]

    # Check if circle - handle as appropriate. Easier of the    #
    # two shapes.                                               #
    if alien.is_circle( ):
        # Check if circle collides with any of the window's     #
        # boundaries. We'll treat them as walls and call the    #
        # does_alien_touch_wall method. Return False if it      #
        # touches one of the walls, since it's considered       #
        # "out of bounds".                                      #
        if does_alien_touch_wall( alien, borders ):
            return False
        
        # At this point we know that the alien does not touch   #
        # the wall. Thus the alien must be inside or outside    #
        # the borders. Use is_point_in_polygon with Centroid    #
        # to determine such. Get the Centroid for such.         #
        centroid = alien.get_centroid( )

        # Check if Centroid is in the borders (treat the window # 
        # as a polygon).                                        #
        if is_point_in_polygon( centroid, borders ):
            return True
        # If the Centroid is not within the polygon, then the   #
        # alien is outside of the window. Return False.         #
        else:
            return False
    # Similar thinking for oblong shapes - we'll use the        #
    # does_alien_touch_wall to determine whether it's on the    # 
    # window, and test both head and tail of the alien with     #
    # is_point_in_polygon to determine whether the alien is in  #
    # the polygon or not. Technically, we only need to test one #
    # point, as does_alien_touch_wall should catch if the alien #
    # crosses a wall / one point is not within the polygon.     #
    else:
        # Check if circle collides with any of the window's     #
        # boundaries. We'll treat them as walls and call the    #
        # does_alien_touch_wall method. Return False if it      #
        # touches one of the walls, since it's considered       #
        # "out of bounds".                                      #
        if does_alien_touch_wall( alien, borders ):
            return False
        
        # At this point we know that the alien does not touch   #
        # the wall. Thus the alien must be inside or outside    #
        # the borders. Use is_point_in_polygon with one of the  #
        # end points to determine such. Get the head for such.  #
        head_and_tail = alien.get_head_and_tail( )        
        head = head_and_tail[ 0 ]

        # Check if end point is in the borders (treat the       #
        # window as a polygon).                                 #
        if is_point_in_polygon( head, borders ):
            return True
        # If the head is not within the polyon, then the alien  #
        # is outside of the window. Return False.               #
        else:
            return False


# --------------------- TO DO (MP5) --------------------------- #
# First, we check if the point lies on any of the edges. If so, #
# then just return.                                             #
# Otherwise, we will turn the point into a line segment that    #
# goes up to the edge of the window - if a point is inside the  #
# given polygon, it will have an odd number of intersections    #
# with the polygon's edges.                                     #
def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """

    # First, check if the point lies on any of the polygon's    #
    # edges.                                                    #
    for i in range( len( polygon ) ):
        # Get first vertex to be checked                        #
        vertex1 = polygon[ i ]
        
        # Check if last edge. We'll hardcode it so the loop     #
        # doesn't break.                                        #
        if i == len( polygon ) - 1:
            vertex2 = polygon[ 0 ]
        else:
            vertex2 = polygon[ i + 1 ]
        
        # Check if point lies on line segment between vertices. #
        if point_segment_distance( point, ( vertex1, vertex2 ) ) == 0:
            return True

    # Point does not lie on any of the polygon's edges. Check   #
    # if it lies inside.                                        #
    # Extend the point into a line segment - an arbitrarily     #
    # large number should be fine. Since the test windows don't #
    # exceed 400, we can just add 500 for good measure.         #
    point_line = ( point, ( 500, point[ 1 ] ) )

    # Check if point_line intersects with any of the polygon's  #
    # edges. Keep track of the number of intersections.         #
    num_intersections = 0
    for i in range( len( polygon ) ):
        # Get first vertex to be checked                        #
        vertex1 = polygon[ i ]

        # Check if last edge. We'll hardcode it so the loop     #
        # doesn't break.                                        #
        if i == len( polygon ) - 1:
            vertex2 = polygon[ 0 ]
        else:
            vertex2 = polygon[ i + 1 ]

        # Check if point_line intersects with given edge.       #
        if do_segments_intersect( point_line, ( vertex1, vertex2 ) ):
            num_intersections += 1
    
    # Check if number of intersections is odd.                  #
    if ( num_intersections % 2 ) == 1:
        return True
    else:
        return False

# --------------------- TO DO (MP5) --------------------------- #
# The roadmap consists of waypoints and edges connecting them;  #
# more precisely, the waypoints and edges that don't run into   #
# obstacles. However, the alien can still manage to touch walls #
# when moving between these waypoints. We need to determine     #
# whether the alien's straight-line path from its current       #
# position touches a wall. There are two scenarios to consider: #
# 1. The alien is ball.                                         #
#    - In this case, we evaluate the path of the alien as a     #
#      line segment between its start and end, treating it like #
#      an oblong that is oriented weirdly. We can check to see  #
#      if the distance of that path from any wall is less than  #
#      the alien's width, as that indicates the alien will      #
#      still manage to collide with a wall.                     #
# 2. The alien is oblong.                                       #
#    - In this case, we evaluate the path of the alien as a     #
#      parallelogram. We can check to see if any walls collide  #
#      with the parallelogram, and whether any walls have their #
#      endpoints within the parallelogram.                      #
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
    # print( 'Moving from', alien.get_centroid( ), 'to', waypoint, 'with circle ==', alien.is_circle( ) )

    # Check if the current position is touching a wall. This    #
    # will probably not happen, but some edge cases may cause   #
    # this to occur so we'll include the check anyways.         #
    if does_alien_touch_wall( alien, walls ):
        return True


    # Check if circle - handle as appropriate. Easier of the    #
    # two shapes.                                               #    
    if alien.is_circle( ):
        # Get start and end waypoints and represent them as one #
        # line segment.                                         #
        start = alien.get_centroid( )
        end = waypoint

        # Check if waypoint is current point. Ignore if so.     #
        if start == end:
            return False

        # Evaluate whether moving the ball to the waypoint via  #
        # a straight path will cause any conditions. Iterate    #
        # through the walls and check if any of them come close #
        # enough to the line segment that is our path. Get the  #
        # width to determine if a wall is too close.            #
        width = alien.get_width( )

        # Check for each potential obstacle or wall.            #
        for wall in walls:
            # Need to format line segments according to         #
            # parameters of segment_distance.                   #
            path = ( start, end )
            obstacle = ( ( wall[ 0 ], wall[ 1 ] ), ( wall[ 2 ], wall[ 3 ] ) )
            # Check if the alien come too close to the wall.    #
            # Return True if the alien collides with a wall,    #
            # i.e. the width comes into contact.                #
            if segment_distance( path, obstacle ) <= width:
                return True
    # Else, is oblong. Evaluate the path as a parallelogram and #
    # determine if any collisions happen with any walls - we    #
    # can test if either endpoint of each wall falls within the #
    # given parallelogram, which indicates that a collision     #
    # occurs.                                                   #
    else:
        # Get start and end waypoints and represent them as one #
        # line segment.                                         #
        start = alien.get_centroid( )
        end = waypoint

        # Check if waypoint is current point. If so, we can't   #
        # ignore bc the current point may be invalid. Check if  #
        # the current point is touching a wall
        if start == end:
            return False

        # Calculate their change in coordinates. We need to     #
        # approximate a parallelogram from the movement of the  #
        # alien, and to do this we need to evaluate where the   #
        # head and tail end up, warranting the need to          #
        # calculate the direction and amount of movement.       #
        dxdy = ( end[ 0 ] - start[ 0 ], end [ 1 ] - start[ 1 ] )

        # To build our parallelogram, we need to get the head   #
        # and the tail of our alien to determine the vertices   #
        # of the parallelogram. We add the change in location   #
        # to our head and tail to get the other end of the      #
        # paralellogram.                                        #
        head_and_tail = alien.get_head_and_tail( )
        start_head = head_and_tail[ 0 ]
        start_tail = head_and_tail[ 1 ]
        end_head   = ( start_head[ 0 ] + dxdy[ 0 ], start_head[ 1 ] + dxdy[ 1 ] )
        end_tail   = ( start_tail[ 0 ] + dxdy[ 0 ], start_tail[ 1 ] + dxdy[ 1 ] )

        # Format start and end heads and tails as a             #
        # parallelogram so we can call is_point_in_polygon      #
        borders = ( start_head, start_tail, end_tail, end_head )

        # For some reason, the code below bugs out when it has  #
        # to deal with horizontal/vertical movement in the same #
        # direction as its orientation. In these cases, the     #
        # movement is NOT represented as a parallelogram, but   #
        # is instead represented as an elongated line segment.  #
        # No change in x indicates that the alien is moving     #
        # vertically. Check for vertical orientation as well.   #
        if dxdy[ 0 ] == 0 and start_head[ 0 ] == start_tail[ 0 ]:
            return handle_directional( alien, walls, dxdy ) 
        # No chane in y indicates that the alien is moving      #
        # horizontally. Check for horizontal orientation as     #
        # well.                                                 #
        if dxdy[ 1 ] == 0 and start_head[ 1 ] == start_tail[ 1 ]:
            return handle_directional( alien, walls, dxdy )
        
        # Iterate through walls and determine if either         #
        # endpoint is in the parallelogram.                     #
        for wall in walls:
            # Get endpoints of wall.                            #
            wall_start = ( wall[ 0 ], wall[ 1 ] )
            wall_end   = ( wall[ 2 ], wall[ 3 ] )

            # Check whether either point falls within the       #
            # parallelogram.                                    #
            if is_point_in_polygon( wall_start, borders ) or is_point_in_polygon( wall_end, borders ):
                return True
            
        # No walls fall within the path of movement. Check if   #
        # the final resting spot will cause any collisions as   #
        # well. Use the width to determine collision.           #
        width = alien.get_width( )
        for wall in walls:
            # Get endpoints of wall.                            #
            wall_start = ( wall[ 0 ], wall[ 1 ] )
            wall_end   = ( wall[ 2 ], wall[ 3 ] )
            
            # Determine if wall collides with end position.     #
            if segment_distance( ( end_head, end_tail ), ( wall_start, wall_end ) ) <= width:
                return True
    
    return False

# Helper function to handle purely horizontal or vertical       #
# movement.                                                     #
def handle_directional(alien: Alien, walls: List[Tuple[int]], dxdy:Tuple[int,int] ):
    """Determines whether the movement in the same direction as the orientation collides with any walls.
 
        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            dxdy (tuple): the change in coordinate between the start and the end of the alien's movement.

        Return:
            True if touched, False if not
    """
    # We have determined the movement is in the same direction  #
    # as the orientation. We need to figure out the new line    #
    # segment we want to construct.                             #
    # Get our starting head/tail and our end head/tail.         #
    head_and_tail = alien.get_head_and_tail( )
    start_head = head_and_tail[ 0 ]
    start_tail = head_and_tail[ 1 ]
    end_head   = ( start_head[ 0 ] + dxdy[ 0 ], start_head[ 1 ] + dxdy[ 1 ] )
    end_tail   = ( start_tail[ 0 ] + dxdy[ 0 ], start_tail[ 1 ] + dxdy[ 1 ] )    

    # Determine the index into dxdy we want to use by           #
    # determining whether we are moving vertically ( dy ) or    #
    # horizontally ( dx ).                                      #
    if dxdy[ 0 ] == 0:
        index = 1
    else:
        index = 0
    
    # We can determine the points by calculating the difference #
    # between the start head and the end tail, or the start     #
    # tail and the end head. We will use the pair that contains #
    # the larger difference.                                    #
    difference_start_head_end_tail = abs( start_head[ index ] - end_tail[ index ] )
    difference_start_tail_end_head = abs( start_tail[ index ] - end_head[ index ] )

    if difference_start_head_end_tail > difference_start_tail_end_head:
        line_segment = ( start_head, end_tail )
    else:
        line_segment = ( start_tail, end_head )

    # Now that we have our line segment, determine if any walls #
    # come within the given width of our alien line segment.    #
    for wall in walls:
        # Get endpoints of wall.                                #
        wall_start = ( wall[ 0 ], wall[ 1 ] )
        wall_end   = ( wall[ 2 ], wall[ 3 ] )

        # Determine if wall is too close to the line segment.   #
        if segment_distance( line_segment, ( wall_start, wall_end ) ) <= alien.get_width( ):
            return True
        
    # None of the walls are too close, no collisions detected.  #
    # Return False to indicate so.                              #
    return False


# --------------------- TO DO (MP5) --------------------------- #
# Compute the Euclidian Distance from a point to a Line Segment #
# defined as the distance from the point to the closest point   #
# on the segment (not a line!)                                  #
def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    # Adopted from some Javascript code, on StackOverflow:      #
    # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment 
    # Is the second most upvoted answer here. Includes a really #
    # helpful visualization behind the reasoning for the        #
    # geometry performed below. For some reason my method       #
    # wasn't working as intended so this works instead.         #
    # The function is modified to fit the needs of the program  #
    # and language.                                             #
    # Essentially, we need to:                                  #
    # 1. Determine if the point lies outside of the two end     #
    #    points, and if so which point it lies closer to        #
    # 2. Calculate based on closest point or line               #

    # print( 'p:', p )
    # print( 's:', s )

    # x, y represent target point                               #
    x = p[ 0 ]
    y = p[ 1 ]
    # x1, y1 to x2, y2 is the line segment                      #
    x1 = s[ 0 ][ 0 ]
    y1 = s[ 0 ][ 1 ]
    x2 = s[ 1 ][ 0 ]
    y2 = s[ 1 ][ 1 ]

    # ( a, b ) is the vector b/w point and base point           #
    a = x - x1
    b = y - y1
    # ( c, d ) is the vector b/w point and end point            #
    c = x2 - x1
    d = y2 - y1

    # Calculate Dot Product and magnitude of our line segment   #
    dot = a * c + b * d
    len_sq = c * c + d * d

    # Param is the equivalent to |a|cos( alpha )
    param = dot / len_sq

    # Case 1: Point is located between the two points           #
    if param < 0:
        xx = x1
        yy = y1
    # Case 2: Point is located outside of the two points,       #
    # closer to end                                             #
    elif param > 1:
        xx = x2
        yy = y2
    # Case 3: Point is located outside of the two points,       #
    # closer to the base
    else:
        xx = x1 + param * c
        yy = y1 + param * d

    # Honestly some wizard math that I don't understand. 
    dx = x - xx
    dy = y - yy 
    return math.sqrt( dx * dx + dy * dy )


# --------------------- TO DO (MP5) --------------------------- #
# Suuuuuper efficient solution presented here:                  #
# https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
# Basically, need to determine if three points are listed in a  #
# Counter Clockwise (CCW) order. Two line segment AB and CD     #
# only intersect iff points A and B are separated by CD.        #
# Then ACD and BCD should have opposite orientation meaning     #
# either ACD or BCD is CCW but not both.                        #
# HOWEVER that solution does NOT take into account collinear    #
# line segments :(. We employ the followin method           #
# constructed by GeeksForGeeks:                                 #
# https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/#
# Same idea, but goes further in depth and spits out the actual #
# orientation of the points to determine if collinearity        #
# exists.                                                       #
def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    # The Line Segments are formatted as 'p1q1' and 'p2q2'.     #
    p1 = s1[ 0 ]
    q1 = s1[ 1 ]
    p2 = s2[ 0 ]
    q2 = s2[ 1 ]

    # Calculate orientations of each set of points.             #
    o1 = orientation( p1, q1, p2 )
    o2 = orientation( p1, q1, q2 )
    o3 = orientation( p2, q2, p1 )
    o4 = orientation( p2, q2, q1 )

    # General Case - if o1 != o2 and o3 != o4 then MUST not     #
    # intersect. Otherwise, we need to check for collinearity.  #
    if o1 != o2 and o3 != o4:
        return True
    
    # == 0 indicates points are collinear. Check if they        #
    # overlap and return True if so.                            #
    if o1 == 0 and collinear_overlap( p1, p2, q1 ):
        return True
    if o2 == 0 and collinear_overlap( p1, q2, q1 ):
        return True
    if o3 == 0 and collinear_overlap( p2, p1, q2 ):
        return True
    if o4 == 0 and collinear_overlap( p2, q1, q2 ):
        return True
    
    # None of the cases - return False.
    return False

# Helper function for do_segments_intersect. See such for more  #
# detail.                                                       #
def orientation(p, q, r):
    """Determines the orientation of points p, q, and r

        Args:
            p: A tuple of coordinates indicating the first  point
            q: A tuple of coordinates indicating the second point
            r: A tuple of coordinates indicating the third  point

        Return:
            0 : Collinear Points
            1 : Clockwise Points
            2: Counterclockwise Points
    """
    # Calculate Orientation. 
    val = ( float( q[ 1 ] - p[ 1 ] ) * float( r[ 0 ] - q[ 0 ] ) ) - ( float( q[ 0 ] - p[ 0 ] ) * float( r[ 1 ] - q[ 1 ] ) ) 
    if val > 0:
        # Clockwise Orientation
        return 1
    elif val < 0:
        # Counterclockwise Orientation
        return 2
    else:
        return 0

# Helper function for do_segments_intersect. See such for more  #
# detail.                                                       #
def collinear_overlap(p, q, r):
    """Determines if three collinear points overlap. Specifically, checks if point q lies on line segment 'pr'

        Args:
            p: A tuple of coordinates representing one end of the line segment
            q: A tuple of coordinates representing the point that may lie on the line segment
            r: A tuple of coordinates representin one end of the line segment

        Return:
            True if line segments intersect, False if not.
    """
    # Determine based off of interval                           #
    if q[ 0 ] <= max( p[ 0 ], r[ 0 ] ) and q[ 0 ] >= min( p[ 0 ], r[ 0 ] ) and q[ 1 ] <= max( p[ 1 ], r[ 1 ] ) and q[ 1 ] >= min( p[ 1 ], r[ 1 ] ):
        return True
    else:
        return False
    
# --------------------- TO DO (MP5) --------------------------- #
# When calculating Segment Distance we have 2 scenarios, of     #
# which the latter can be broken down into 4 different cases,   #
# to consider:                                                  #
# 1. The Segments Intersect                                     #
#    - This is easy, as their distance will be 0                #
# 2. The Segments Do Not Intersect                              #
#    - This indicates that the closest points between two       #
#      segments will one line segment's end point to the other  #
#      line segment ( just visualize it in your head ). Thus,   #
#      we need to consider each end point's distance to the     #
#      other line segment.                                      #
def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """

    # If the Segments Intersect, the distance returned is 0.    #
    if do_segments_intersect( s1, s2 ):
        return 0

    # Otherwise, we gather the distance of each endpoint to the #
    # other segment and return the minimum distance.            #
    distances = [ ]

    # Consider endpoints of s1.                                 #
    distances.append( point_segment_distance( s1[ 0 ], s2 ) )
    distances.append( point_segment_distance( s1[ 1 ], s2 ) )

    # Consider the endpoints of s2.                             #
    distances.append( point_segment_distance( s2[ 0 ], s1 ) )
    distances.append( point_segment_distance( s2[ 1 ], s1 ) )
    
    # Return the minimum of the distances. Takes into account   #
    # parallel lines, since all distances will be the same if   #
    # lines are parallel.                                       #
    return min( distances )


# --------------------- TO DO (MP5) --------------------------- #
# Minor Assertions provided by Course Staff to help with        #
# debugging. Execute with: python geometry.py                   #
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
"""
The following module provides functions for manipulating bounding box coordinates. These functions help convert bounding box coordinates between different formats (absolute and relative) and calculate the corner coordinates of a bounding box from its center and dimensions.

Functions included:
1. get_corners: Converts relative bounding box coordinates to absolute corner coordinates.
2. from_corners_to_relative: Converts absolute corner coordinates to relative bounding box coordinates.

"""

def get_corners(x_center:float, y_center:float, width:float, height:float, 
                img_width:int, img_height:int) -> list[list]:

    """
    This function returns the absolute coordinates of the four corners of aa bounding box, in a list,
    from its relative coordinates (x,y,w,h).

    :param x_center: 
        - Type: float
        - Description: Relative x coordinate of the center of the bounding box. Value should be between 0 and 1.

    :param y_center: 
        - Type: float
        - Description: Relative y coordinate of the center of the bounding box. Value should be between 0 and 1.

    :param width: 
        - Type: float
        - Description: Relative width of the bounding box. Value should be between 0 and 1, relative to image width.

    :param height: 
        - Type: float
        - Description: Relative height of the bounding box. Value should be between 0 and 1, relative to image height.

    :param img_width: 
        - Type: int
        - Description: The width of the image in pixels.

    :param img_height: 
        - Type: int
        - Description: The height of the image in pixels.

    :return: 
        - Type: list of lists
        - Description: A list of four lists, each containing the [x, y] coordinates of a corner of the bounding box
                       in the order [upper_left, upper_right, bottom_right, bottom_left].
    """

    # Compute the exact (unrounded) width and height of the bounding box in pixels
    abs_width = float(width) * int(img_width)
    abs_height = float(height) * int(img_height)

    #  Calculate the absolute pixel coordinates of the upper-left corner of the bounding box
    upper_left_x = round((float(x_center) * int(img_width)) - abs_width / 2)
    upper_left_y = round((float(y_center) * int(img_height)) - abs_height / 2)

    # Round width and height only after computing the upper-left corner, so that it is based on exact values
    abs_width, abs_height = round(abs_width), round(abs_height)
    
    # Get the absolute pixel coordinates of each corner of the bounding box
    upper_left = [upper_left_x, upper_left_y]
    upper_right = [upper_left_x + abs_width, upper_left_y]
    bottom_right = [upper_left_x + abs_width, upper_left_y + abs_height]
    bottom_left = [upper_left_x, upper_left_y + abs_height]

    corners = [upper_left, upper_right, bottom_right, bottom_left]
    
    # print(corners)
    return corners

def from_corners_to_relative(new_upper_left, new_bottom_right, TP_img_width, TP_img_height):
    """
    This  function converts coordinates of the upper left and bottom right corners 
    of a bounding box to relative coordinates with respect to the dimensions of an image. 
    The function returns the center of the bounding box (transformed_x_center, transformed_y_center) 
    and its width and height (transformed_width, transformed_height) relative to the dimensions 
    of the transformed image.
    

    :param new_upper_left: 
        - Type: list or tuple of int
        - Description: Absolute coordinates of the upper-left corner of the bounding box [x, y].

    :param new_bottom_right: 
        - Type: list or tuple of int
        - Description: Absolute coordinates of the bottom-right corner of the bounding box [x, y].

    :param TP_img_width: 
        - Type: int
        - Description: Width of the image in pixels.

    :param TP_img_height: 
        - Type: int
        - Description: Height of the image in pixels.

    :return: 
        - Type: tuple
        - Description: A tuple containing:
                       - transformed_x_center: Relative x coordinate of the center of the bounding box.
                       - transformed_y_center: Relative y coordinate of the center of the bounding box.
                       - transformed_width: Relative width of the bounding box.
                       - transformed_height: Relative height of the bounding box.
    """

    # Convert corner coordinates into relative ones with respect to the dimensions of the transformed image
    transformed_x_center = (new_upper_left[0] + new_bottom_right[0]) / 2 / TP_img_width
    transformed_y_center = (new_upper_left[1] + new_bottom_right[1]) / 2 / TP_img_height
    transformed_width = (new_bottom_right[0] - new_upper_left[0]) / TP_img_width
    transformed_height = (new_bottom_right[1] - new_upper_left[1]) / TP_img_height

    # print(transformed_x_center, transformed_y_center, transformed_width, transformed_height)
    return transformed_x_center, transformed_y_center, transformed_width, transformed_height
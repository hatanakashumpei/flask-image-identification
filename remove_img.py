# coding: utf-8

"""
this code is to remove images by using opencv
@ shumpei hatanaka
"""

import os
import sys
import time
import glob
ROS_MELODIC_PATH = '/opt/ros/melodic/lib/python2.7/dist-packages'
ROS_KINETIC_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ROS_MELODIC_PATH in sys.path:
    sys.path.remove(ROS_MELODIC_PATH)
if ROS_KINETIC_PATH in sys.path:
    sys.path.remove(ROS_KINETIC_PATH)
import cv2


def get_file_list():
    """Obtain img file list

    Returns:
        list: list of file path
    """
    return list(glob.glob("./data/*/*"))


def delete_img(path):
    """delete img

    Args:
        path (str): img path
    """
    os.remove(path)


def open_img(path):
    """Visually check to see whether to save the image or not.

    Args:
        path (str): img path

    Returns:
        bool: whether to save the image or not (default : True)
    """
    save_img = True
    img = cv2.imread(path, 1)
    while True:
        cv2.imshow(path, img)
        if cv2.waitKey(1) & 0xFF == ord("s"):
            break
        if cv2.waitKey(1) & 0xFF == ord("d"):
            save_img = False
            break
    cv2.destroyAllWindows()

    return save_img


def main():
    """Run remove.py
    """
    img_list = get_file_list()
    for img_path in img_list:
        time.sleep(0.5)
        save_img = open_img(img_path)
        if not save_img:
            delete_img(img_path)


if __name__ == '__main__':
    main()

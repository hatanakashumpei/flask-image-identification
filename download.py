# coding: utf-8

"""
this code is to download images
@ shumpei hatanaka
"""

import os
import time
import sys
from urllib import request
from urllib.request import urlretrieve
from flickrapi import FlickrAPI


# PARAM
PARAM = {
    "KEY": "6c56cfb6290cab4ab7976f1f0f611e4f",
    "SECRET": "0643c6d81d4e7569",
    "WAITTIME": 1
}


def get_flickrapi(name):
    """get image url by using FlickrAPI

    Args:
        name (str): search word

    Returns:
        photos : list of image info
    """
    flickr = FlickrAPI(PARAM["KEY"], PARAM["SECRET"], format='parsed-json')

    result = flickr.photos.search(
        text=name,
        per_page=400,
        media='photos',
        sort='relevance',
        safe_search=1,
        extras='url_q, license'
    )
    photos = result['photos']['photo']
    return photos


def dowonload_img(photos, path):
    """download images

    Args:
        photos (list): list of image info
        path (str): save image path
    """
    for i, photo in enumerate(photos):
        print(f"{i + 1} / {len(photos)}")
        print(photo)
        url_q = photo['url_q']
        file_path = f"{path}/{photo['id']}.jpg"
        if not os.path.exists(file_path):
            urlretrieve(url_q, file_path)
            time.sleep(PARAM["WAITTIME"])


def main(args):
    """run download.py

    Args:
        args (list): sys.argv
    """
    search_name = args[1]
    save_dir = f"./data/{search_name}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    photos = get_flickrapi(search_name)
    dowonload_img(photos, save_dir)


if __name__ == '__main__':
    args = sys.argv
    main(args)

"""
This module contains functions to handle file operations.
"""

import json
import os


def create_dir(path):
    """
    Creates a directory if it does not exist.

    Args:
        path: Path to create the directory.
    Returns:
        bool: True if the directory was created or already exists.
    """
    # checking if the directory demo_folder2
    # exist or not.
    if not os.path.isdir(path):
        # if the demo_folder2 directory is
        # not present then create it.
        os.makedirs(path)
    return True


def __load_json__(path):
    """
    Loads a JSON file.

    Args:
        path: Path to the JSON file.
    Returns:
        dict: Dictionary containing the JSON data.
    """
    with open(path, "r", encoding="utf-8") as f:
        tmp = json.loads(f.read())

    return tmp


def join_path(path, file):
    """
    Joins a path and a file.

    Args:
        path: Path to join.
        file: File to join.
    Returns:
        str: Joined path.
    """
    if path.endswith("/") and file.startswith("/"):
        path = f"{path}{file[1:]}"
    if path.endswith("/"):
        path = f"{path}{file}"
    if file.startswith("/"):
        path = f"{path}{file}"
    else:
        path = f"{path}/{file}"

    return path

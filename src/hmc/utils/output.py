
import json


def save_dict_to_json(dictionary, file_path):
    """
    Saves a dictionary to a JSON file.

    Args:
        dictionary (dict): The dictionary to be saved.
        file_path (str): The path to the JSON file where the dictionary will be saved.

    Raises:
        TypeError: If the dictionary contains non-serializable objects.
        OSError: If the file cannot be written.

    Example:
        save_dict_to_json({'a': 1, 'b': 2}, 'output.json')
    """
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(dictionary, json_file, ensure_ascii=False, indent=4)


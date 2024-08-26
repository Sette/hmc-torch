
import os
import json
import datetime

def create_dir(path):
    # checking if the directory demo_folder2
    # exist or not.
    if not os.path.isdir(path):
        # if the demo_folder2 directory is
        # not present then create it.
        os.makedirs(path)
    return True

def __load_json__(path):
    with open(path, 'r') as f:
        tmp = json.loads(f.read())

    return tmp

def join_path(path, file):
    if path.endswith("/") and file.startswith("/"):
        return "{}{}".format(path, file[1:])
    elif path.endswith("/"):
        return "{}{}".format(path, file)
    elif file.startswith("/"):
        return "{}{}".format(path, file)
    else:
        return "{}/{}".format(path, file)


def create_job_id():
    # Obtém a data e hora atual
    now = datetime.datetime.now()
    # Formata a data e hora no formato YYYYMMDD_HHMMSS
    job_id = now.strftime("%Y%m%d_%H%M%S")
    return job_id

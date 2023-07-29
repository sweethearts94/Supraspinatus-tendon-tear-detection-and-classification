import os
from queue import Queue, Empty


def recursion_folder(root_path: str, cal_sufflx: str):
    filepath_set = []
    temp_q = Queue()
    [temp_q.put(os.path.join(root_path, filepath)) for filepath in os.listdir(root_path)]
    while True:
        try:
            target_path = temp_q.get_nowait()
            if os.path.isdir(target_path):
                filepath_set += recursion_folder(target_path, cal_sufflx)
            if target_path.endswith(cal_sufflx.lower()) or target_path.endswith(cal_sufflx.upper()):
                filepath_set.append(target_path)
        except Empty:
            break
    return filepath_set

def check_and_create_folder(path: tuple) -> bool:
    try:
        path_str = os.path.join(*path)
        if not os.path.exists(path_str):
            os.makedirs(path_str)
    except Exception:
        return False
    return True
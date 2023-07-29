import os
import time

class Logger:
    def __init__(self, logfile_name: str, path: tuple = ("log",), buffer_size: int = 3):
        path_list = list(path) + ["{}.log".format(logfile_name)]
        self.__filepath = os.path.join(*path_list)
        self.__write_buffer = []
        self.__buffer_size = buffer_size
        self.__write_lock = False

    def info(self, msg: str, write_directly: bool = False):
        template = "{}\n".format(msg)
        if write_directly:
            retry_time = 0
            while self.__write_lock:
                time.sleep(1)
                retry_time += 1
                if retry_time >= 30:
                    break
            self.__write_lock = True
            with open(self.__filepath, "a+", 1) as log_handle:
                log_handle.write(template)
            self.__write_lock = False
        else:
            if len(self.__write_buffer) >= self.__buffer_size:
                with open(self.__filepath, "a+", 1) as log_handle:
                    log_handle.write("".join(self.__write_buffer))
                self.__write_buffer = []
            else:
                self.__write_buffer.append(template)
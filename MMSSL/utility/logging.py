import os
from datetime import datetime


def mk_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


class Logger:
    def __init__(self, filename, is_debug, path="/home/ww/Code/work5/MMSSL/logs/"):
        self.filename = filename
        self.path = path + "/logs"
        self.log_ = not is_debug
        mk_dir(self.path)

    def logging(self, s):
        s = str(s)
        print(datetime.now().strftime("%Y-%m-%d %H:%M: "), s)
        if self.log_:
            with open(
                os.path.join(os.path.join(self.path, self.filename)), "a+"
            ) as f_log:
                f_log.write(
                    str(datetime.now().strftime("%Y-%m-%d %H:%M:  ")) + s + "\n"
                )

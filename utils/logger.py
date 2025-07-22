import logging
import time

class SimpleLogger:
    def __init__(self, log_file=None):
        try:
            self.logger = logging.getLogger('FASTLogger')
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            if log_file:
                fh = logging.FileHandler(log_file)
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        except Exception as e:
            print(f"[ERROR] Logger initialization error: {e}")
    def info(self, msg):
        try:
            self.logger.info(msg)
        except Exception as e:
            print(f"[ERROR] Logger info error: {e}")
    def timeit(self, msg):
        try:
            start = time.time()
            yield
            end = time.time()
            self.info(f'{msg} - Time: {end-start:.3f}s')
        except Exception as e:
            print(f"[ERROR] Logger timeit error: {e}") 
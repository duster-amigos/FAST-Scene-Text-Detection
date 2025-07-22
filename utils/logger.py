import logging
import time

class SimpleLogger:
    """Simple logger for printing and saving logs to file. Also supports timing code blocks."""
    def __init__(self, log_file=None):
        """Initializes logger. If log_file is given, logs are saved to file as well as printed."""
        try:
            # Set up logger with INFO level
            self.logger = logging.getLogger('FASTLogger')
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            # Log to file if specified
            if log_file:
                fh = logging.FileHandler(log_file)
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)
            # Always log to console
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        except Exception as e:
            print(f"[ERROR] Logger initialization error: {e}")
    def info(self, msg):
        """Logs a message at INFO level."""
        try:
            self.logger.info(msg)
        except Exception as e:
            print(f"[ERROR] Logger info error: {e}")
    def timeit(self, msg):
        """Context manager for timing a code block. Usage: with logger.timeit('desc'): ..."""
        try:
            # Start timer
            start = time.time()
            yield
            # End timer and log elapsed time
            end = time.time()
            self.info(f'{msg} - Time: {end-start:.3f}s')
        except Exception as e:
            print(f"[ERROR] Logger timeit error: {e}") 
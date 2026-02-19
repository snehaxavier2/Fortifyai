import os
from datetime import datetime


class PreprocessLogger:
    def __init__(self, log_path):
        """
        log_path: Full path to log file
        """
        self.log_path = log_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Initialize log file
        with open(self.log_path, "a") as f:
            f.write("\n")
            f.write("=" * 60 + "\n")
            f.write(f"New Session Started: {self._timestamp()}\n")
            f.write("=" * 60 + "\n")

    def _timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def info(self, message):
        formatted_message = f"[INFO] {self._timestamp()} - {message}"
        self._write(formatted_message)

    def error(self, message):
        formatted_message = f"[ERROR] {self._timestamp()} - {message}"
        self._write(formatted_message)

    def summary(self, message):
        formatted_message = f"[SUMMARY] {self._timestamp()} - {message}"
        self._write(formatted_message)

    def _write(self, message):
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

import logging

class FileFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    green = "\x1b[32;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_debug = "%(asctime)s - %(levelname)s    - %(filename)s:%(lineno)d - %(message)s"
    format_info = "%(asctime)s - %(levelname)s     - %(filename)s:%(lineno)d - %(message)s"
    format_warning = "%(asctime)s - %(levelname)s  - %(filename)s:%(lineno)d - %(message)s"
    format_error = "%(asctime)s - %(levelname)s    - %(filename)s:%(lineno)d - %(message)s"
    format_critical = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

    FORMATS = {
        logging.DEBUG: format_debug,
        logging.INFO: format_info,
        logging.WARNING: format_warning,
        logging.ERROR: format_error,
        logging.CRITICAL: format_critical
    }

    def format(
        self, 
        record,
    ):
        """
        Format the log record
        Args:
            record: log record
        Returns:
            formatted log record
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    

class TerminalFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    green = "\x1b[32;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_debug = "%(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    format_info = "%(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    format_warning = "%(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    format_error = "%(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    format_critical = "%(levelname)s - %(filename)s:%(lineno)d - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_debug + reset,
        logging.INFO: green + format_info + reset,
        logging.WARNING: yellow + format_warning + reset,
        logging.ERROR: red + format_error + reset,
        logging.CRITICAL: bold_red + format_critical + reset
    }

    def format(
        self, 
        record,
    ):
        """
        Format the log record
        Args:
            record: log record
        Returns:
            formatted log record
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
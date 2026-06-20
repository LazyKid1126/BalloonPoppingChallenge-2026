import logging


class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',   # Blue
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[95m' # Purple
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')

        asctime = f"{self.formatTime(record, self.datefmt)}.{int(record.msecs):03d}"
        levelname = f"{color}{record.levelname}{self.RESET}"
        name = record.name
        message = f"{color}{record.getMessage()}{self.RESET}"

        return f"{asctime} [{levelname}] {name}: {message}"

def setup_logging(level=logging.INFO):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    ))

    logging.basicConfig(
        level=level,
        handlers=[console_handler]
    )

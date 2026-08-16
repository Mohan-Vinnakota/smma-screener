import logging
import os
from datetime import datetime
from config import LOG_FOLDER

os.makedirs(LOG_FOLDER, exist_ok=True)
log_file = f"{LOG_FOLDER}/smma_{datetime.now().strftime('%Y%m%d')}.log"

# Our app logger
logger = logging.getLogger("smma")
logger.setLevel(logging.INFO)

# File handler — clean format, no colors
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S"
))

# Terminal handler
terminal_handler = logging.StreamHandler()
terminal_handler.setLevel(logging.INFO)
terminal_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S"
))

logger.addHandler(file_handler)
logger.addHandler(terminal_handler)

# Stop Flask and WebSocket logs from mixing in
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("websockets").setLevel(logging.ERROR)
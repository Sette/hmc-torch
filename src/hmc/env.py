"""
Environment and logging configuration module.

This module manages the configuration of logging verbosity levels using
environment variables. It defines a global log level (GLOBAL_LOG_LEVEL)
and allows configuring specific levels for different components of the
system, such as training (TRAIN), model (MODEL), and dataset (DATASET).
"""

import logging
import os
import sys

log_levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]

GLOBAL_LOG_LEVEL = os.environ.get("GLOBAL_LOG_LEVEL", "").upper()
if GLOBAL_LOG_LEVEL in log_levels:
    logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL, force=True)
else:
    GLOBAL_LOG_LEVEL = "INFO"

log = logging.getLogger(__name__)
log.info("GLOBAL_LOG_LEVEL: %s", GLOBAL_LOG_LEVEL)


log_sources = [
    "TRAIN",
    "MODEL",
    "DATASET",
]

SRC_LOG_LEVELS = {}

for source in log_sources:
    LOG_ENV_VAR = source + "_LOG_LEVEL"
    SRC_LOG_LEVELS[source] = os.environ.get(LOG_ENV_VAR, "").upper()
    if SRC_LOG_LEVELS[source] not in log_levels:
        SRC_LOG_LEVELS[source] = GLOBAL_LOG_LEVEL
    log.info("%s: %s", LOG_ENV_VAR, SRC_LOG_LEVELS[source])

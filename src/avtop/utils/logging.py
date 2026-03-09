# -*- coding: utf-8 -*-
# src/avtop/utils/logging.py
import logging, os, sys
from typing import Optional

_LOGGERS = {}

def get_logger(name: str = "avtop", level: Optional[str] = None) -> logging.Logger:
    """统一日志器：
    - 默认级别 INFO；若环境变量 AVTOP_DEBUG=1 或 level='DEBUG' 则 DEBUG
    - 控制台格式包含级别、模块、行号，便于快速定位
    """
    global _LOGGERS
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # level 解析
    env_debug = os.environ.get("AVTOP_DEBUG", "0") == "1"
    level = (level or ("DEBUG" if env_debug else "INFO")).upper()
    lvl = getattr(logging, level, logging.INFO)
    logger.setLevel(lvl)

    handler = logging.StreamHandler(stream=sys.stdout)
    fmt = "[%(levelname)s] %(asctime)s %(name)s:%(lineno)d | %(message)s"
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%H:%M:%S"))
    logger.addHandler(handler)

    _LOGGERS[name] = logger
    return logger

import time
import psutil
import os
from functools import wraps

from dataclasses import is_dataclass, replace

def measure_performance(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        process = psutil.Process(os.getpid())

        mem_before = process.memory_info().rss
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start
        mem_after = process.memory_info().rss

        mem_delta = mem_after - mem_before

        meta = {
            "elapsed_seconds": elapsed,
            "memory_delta_bytes": mem_delta
        }

        # ----------------------------------------------------
        # SAFE ATTACHMENT LOGIC (DICT OR DATACLASS)
        # ----------------------------------------------------

        if isinstance(result, dict):
            result.setdefault("meta", {})
            result["meta"].update(meta)

        elif (
            is_dataclass(result)
            and hasattr(result, "performance")
        ):
            result = replace(
                result,
                performance=meta
            )

        elif hasattr(result, "metadata"):
            result.metadata["performance"] = meta

        elif hasattr(result, "performance"):
            result.performance = meta

        return result

    return wrapper

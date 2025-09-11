# Kalico compat for ToolProbe
# Loads only on Kalico; otherwise your outer loader falls back to base ToolProbe.

import importlib.util

def _assert_kalico(printer):
    # Kalico ships dockable_probe; upstream Klipper usually doesn't.
    if importlib.util.find_spec("klippy.extras.dockable_probe") is None:
        raise ImportError("Not Kalico (dockable_probe module missing)")

def is_kalico(printer) -> bool:
    try:
        _assert_kalico(printer)
        return True
    except ImportError:
        return False

from .. import tool_probe as base  # your original module

class ToolProbe(base.ToolProbe):
    def __init__(self, config):
        _assert_kalico(config.get_printer())
        super().__init__(config)

def load_config_prefix(config):
    _assert_kalico(config.get_printer())
    return ToolProbe(config)

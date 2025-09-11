# Kalico compat for ToolsCalibrate (multi-axis X/Y/Z)
# Loads only on Kalico; otherwise your outer loader falls back to base ToolsCalibrate.

import importlib.util

def _assert_kalico(printer):
    # Kalico ships a dockable_probe extra; upstream Klipper usually doesn't.
    if importlib.util.find_spec("klippy.extras.dockable_probe") is None:
        raise ImportError("Not Kalico (dockable_probe module missing)")

def is_kalico(printer) -> bool:
    try:
        _assert_kalico(printer)
        return True
    except ImportError:
        return False

from .. import tools_calibrate as base  # your original module

class ToolsCalibrate(base.ToolsCalibrate):
    """
    Kalico flavor. Start identical; override here only if Kalico drifts.
    We keep your command surface and probing logic intact.
    """
    def __init__(self, config):
        _assert_kalico(config.get_printer())
        super().__init__(config)
        # If you later want to hook Kalico z_calibration, do it here:
        # zcal = self.printer.lookup_object('z_calibration', default=None)
        # if zcal: ... integrate optionally without changing your commands.

def load_config(config):
    _assert_kalico(config.get_printer())
    return ToolsCalibrate(config)

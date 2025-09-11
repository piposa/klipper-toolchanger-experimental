# Kalico compat for ToolProbeEndstop
# Only difference: don't seize 'probe' if one already exists (e.g. [dockable_probe]).

import importlib.util

def _assert_kalico(printer):
    if importlib.util.find_spec("klippy.extras.dockable_probe") is None:
        raise ImportError("Not Kalico (dockable_probe module missing)")

def is_kalico(printer) -> bool:
    try:
        _assert_kalico(printer)
        return True
    except ImportError:
        return False

from .. import tool_probe_endstop as base
from .. import probe as probe  # reuse your helpers

class ToolProbeEndstop(base.ToolProbeEndstop):
    def __init__(self, config):
        _assert_kalico(config.get_printer())
        # Rebuild init without force-claiming 'probe'
        self.printer = config.get_printer()
        self.reactor = self.printer.get_reactor()
        self.name = config.get_name()
        self.tool_probes = {}
        self.last_query = {}  # tool -> endstop state
        self.active_probe = None
        self.active_tool_number = -1
        self.gcode_macro = self.printer.load_object(config, 'gcode_macro')
        self.crash_detection_active = False
        self.crash_lasttime = 0.0

        # Your router + helpers
        self.mcu_probe = base.EndstopRouter(self.printer)
        self.param_helper = probe.ProbeParameterHelper(config)
        self.homing_helper = probe.HomingViaProbeHelper(
            config, self.mcu_probe, self.param_helper
        )
        self.probe_session = probe.ProbeSessionHelper(
            config, self.param_helper, self.homing_helper.start_probe_session
        )

        # If no global 'probe' exists, behave like your base class (register PROBE/QUERY_PROBE)
        self._we_are_probe_provider = self.printer.lookup_object('probe', default=None) is None
        if self._we_are_probe_provider:
            # Create the command helper (registers PROBE/QUERY_PROBE etc.)
            self.cmd_helper = probe.ProbeCommandHelper(
                config, self, self.mcu_probe.query_endstop
            )
            # Provide the 'probe' object
            self.printer.add_object('probe', self)
        else:
            # Kalico already has a probe (likely [dockable_probe]); coexist quietly
            self.cmd_helper = None

        # Crash handling bits from your base
        self.crash_mintime = config.getfloat('crash_mintime', 0.5, above=0.0)
        self.crash_gcode = self.gcode_macro.load_template(config, 'crash_gcode', '')
        self.printer.register_event_handler("klippy:connect", self._handle_connect)

        # Always expose your unique commands
        self.gcode = self.printer.lookup_object('gcode')
        self.gcode.register_command(
            'SET_ACTIVE_TOOL_PROBE', self.cmd_SET_ACTIVE_TOOL_PROBE,
            desc=self.cmd_SET_ACTIVE_TOOL_PROBE_help
        )
        self.gcode.register_command(
            'DETECT_ACTIVE_TOOL_PROBE', self.cmd_DETECT_ACTIVE_TOOL_PROBE,
            desc=self.cmd_DETECT_ACTIVE_TOOL_PROBE_help
        )
        self.gcode.register_command(
            'START_TOOL_PROBE_CRASH_DETECTION', self.cmd_START_TOOL_PROBE_CRASH_DETECTION,
            desc=self.cmd_START_TOOL_PROBE_CRASH_DETECTION_help
        )
        self.gcode.register_command(
            'STOP_TOOL_PROBE_CRASH_DETECTION', self.cmd_STOP_TOOL_PROBE_CRASH_DETECTION,
            desc=self.cmd_STOP_TOOL_PROBE_CRASH_DETECTION_help
        )

def load_config(config):
    _assert_kalico(config.get_printer())
    return ToolProbeEndstop(config)

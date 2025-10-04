"""Backport probe helpers missing from Kalico."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, List

_LOGGER = logging.getLogger(__name__)


def _try_import(*names: str):
    for name in names:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def _ensure(probe_mod: Any, name: str, factory: Callable[[], Any], changed: List[str]):
    if not hasattr(probe_mod, name):
        setattr(probe_mod, name, factory())
        changed.append(name)


def _calc_probe_z_average_factory():
    def calc_probe_z_average(positions, method="average"):
        if method != "median":
            count = float(len(positions))
            return [sum(pos[i] for pos in positions) / count for i in range(3)]
        z_sorted = sorted(positions, key=lambda p: p[2])
        middle = len(positions) // 2
        if len(positions) & 1:
            return z_sorted[middle]
        return calc_probe_z_average(z_sorted[middle - 1 : middle + 1], "average")

    return calc_probe_z_average


def _lookup_minimum_z_factory(manual_probe_mod):
    def lookup_minimum_z(config):
        if hasattr(manual_probe_mod, "lookup_z_endstop_config"):
            zconfig = manual_probe_mod.lookup_z_endstop_config(config)
        elif hasattr(config, "has_section") and config.has_section("stepper_z"):
            zconfig = config.getsection("stepper_z")
        else:
            zconfig = None
        if zconfig is not None:
            return zconfig.getfloat("position_min", 0.0, note_valid=False)
        pconfig = config.getsection("printer")
        return pconfig.getfloat("minimum_z_position", 0.0, note_valid=False)

    return lookup_minimum_z


def _lookup_z_steppers_factory():
    class LookupZSteppers:
        def __init__(self, config, add_stepper_cb):
            self.printer = config.get_printer()
            self.add_stepper_cb = add_stepper_cb
            self.printer.register_event_handler(
                "klippy:mcu_identify", self._handle_mcu_identify
            )

        def _handle_mcu_identify(self):
            kin = self.printer.lookup_object("toolhead").get_kinematics()
            for stepper in kin.get_steppers():
                if stepper.is_active_axis("z"):
                    self.add_stepper_cb(stepper)

    return LookupZSteppers


def _run_single_probe_factory(probe_mod):
    def run_single_probe(probe, gcmd):
        probe_session = probe.start_probe_session(gcmd)
        probe_session.run_probe(gcmd)
        pos = probe_session.pull_probed_results()[0]
        probe_session.end_probe_session()
        return pos

    return run_single_probe


def _probe_parameter_helper_factory():
    class ProbeParameterHelper:
        def __init__(self, config):
            gcode = config.get_printer().lookup_object("gcode")
            self.dummy_gcode_cmd = gcode.create_gcode_command("", "", {})
            self.speed = config.getfloat("speed", 5.0, above=0.0)
            self.lift_speed = config.getfloat("lift_speed", self.speed, above=0.0)
            self.sample_count = config.getint("samples", 1, minval=1)
            self.sample_retract_dist = config.getfloat(
                "sample_retract_dist", 2.0, above=0.0
            )
            atypes = ["median", "average"]
            self.samples_result = config.getchoice(
                "samples_result", atypes, "average"
            )
            self.samples_tolerance = config.getfloat(
                "samples_tolerance", 0.100, minval=0.0
            )
            self.samples_retries = config.getint(
                "samples_tolerance_retries", 0, minval=0
            )

        def get_probe_params(self, gcmd=None):
            if gcmd is None:
                gcmd = self.dummy_gcode_cmd
            probe_speed = gcmd.get_float("PROBE_SPEED", self.speed, above=0.0)
            lift_speed = gcmd.get_float("LIFT_SPEED", self.lift_speed, above=0.0)
            samples = gcmd.get_int("SAMPLES", self.sample_count, minval=1)
            sample_retract_dist = gcmd.get_float(
                "SAMPLE_RETRACT_DIST", self.sample_retract_dist, above=0.0
            )
            samples_tolerance = gcmd.get_float(
                "SAMPLES_TOLERANCE", self.samples_tolerance, minval=0.0
            )
            samples_retries = gcmd.get_int(
                "SAMPLES_TOLERANCE_RETRIES", self.samples_retries, minval=0
            )
            samples_result = gcmd.get("SAMPLES_RESULT", self.samples_result)
            return {
                "probe_speed": probe_speed,
                "lift_speed": lift_speed,
                "samples": samples,
                "sample_retract_dist": sample_retract_dist,
                "samples_tolerance": samples_tolerance,
                "samples_tolerance_retries": samples_retries,
                "samples_result": samples_result,
            }

    return ProbeParameterHelper


def _probe_offsets_helper_factory():
    class ProbeOffsetsHelper:
        def __init__(self, config):
            self.x_offset = config.getfloat("x_offset", 0.0)
            self.y_offset = config.getfloat("y_offset", 0.0)
            self.z_offset = config.getfloat("z_offset")

        def get_offsets(self):
            return self.x_offset, self.y_offset, self.z_offset

    return ProbeOffsetsHelper


def _probe_session_helper_factory(probe_mod):
    calc_probe_z_average = probe_mod.calc_probe_z_average
    HINT_TIMEOUT = getattr(probe_mod, "HINT_TIMEOUT", "")

    class ProbeSessionHelper:
        def __init__(self, config, param_helper, start_session_cb):
            self.printer = config.get_printer()
            self.param_helper = param_helper
            self.start_session_cb = start_session_cb
            self.hw_probe_session = None
            self.results = []
            self.printer.register_event_handler(
                "gcode:command_error", self._handle_command_error
            )

        def _handle_command_error(self):
            if self.hw_probe_session is not None:
                try:
                    self.end_probe_session()
                except Exception:
                    logging.exception("Multi-probe end")

        def _probe_state_error(self):
            raise self.printer.command_error(
                "Internal probe error - start/end probe session mismatch"
            )

        def start_probe_session(self, gcmd):
            if self.hw_probe_session is not None:
                self._probe_state_error()
            self.hw_probe_session = self.start_session_cb(gcmd)
            self.results = []
            return self

        def end_probe_session(self):
            hw_probe_session = self.hw_probe_session
            if hw_probe_session is None:
                self._probe_state_error()
            self.results = []
            self.hw_probe_session = None
            hw_probe_session.end_probe_session()

        def _probe(self, gcmd):
            toolhead = self.printer.lookup_object("toolhead")
            curtime = self.printer.get_reactor().monotonic()
            if "z" not in toolhead.get_status(curtime)["homed_axes"]:
                raise self.printer.command_error("Must home before probe")
            try:
                self.hw_probe_session.run_probe(gcmd)
                epos = self.hw_probe_session.pull_probed_results()[0]
            except self.printer.command_error as err:
                reason = str(err)
                if "Timeout during endstop homing" in reason:
                    reason += HINT_TIMEOUT
                raise self.printer.command_error(reason)
            self.printer.send_event("probe:update_results", epos)
            gcode = self.printer.lookup_object("gcode")
            gcode.respond_info(
                "probe at %.3f,%.3f is z=%.6f" % (epos[0], epos[1], epos[2])
            )
            return epos[:3]

        def run_probe(self, gcmd):
            if self.hw_probe_session is None:
                self._probe_state_error()
            params = self.param_helper.get_probe_params(gcmd)
            toolhead = self.printer.lookup_object("toolhead")
            probexy = toolhead.get_position()[:2]
            retries = 0
            positions = []
            sample_count = params["samples"]
            while len(positions) < sample_count:
                pos = self._probe(gcmd)
                positions.append(pos)
                z_positions = [p[2] for p in positions]
                if max(z_positions) - min(z_positions) > params["samples_tolerance"]:
                    if retries >= params["samples_tolerance_retries"]:
                        raise gcmd.error(
                            "Probe samples exceed samples_tolerance"
                        )
                    gcmd.respond_info(
                        "Probe samples exceed tolerance. Retrying..."
                    )
                    retries += 1
                    positions = []
                if len(positions) < sample_count:
                    toolhead.manual_move(
                        probexy
                        + [pos[2] + params["sample_retract_dist"]],
                        params["lift_speed"],
                    )
            epos = calc_probe_z_average(positions, params["samples_result"])
            self.results.append(epos)

        def pull_probed_results(self):
            res = self.results
            self.results = []
            return res

    return ProbeSessionHelper


def _homing_via_probe_helper_factory(probe_mod, pins_mod):
    lookup_minimum_z = probe_mod.lookup_minimum_z
    LookupZSteppers = probe_mod.LookupZSteppers

    class HomingViaProbeHelper:
        def __init__(self, config, mcu_probe, param_helper):
            self.printer = config.get_printer()
            self.mcu_probe = mcu_probe
            self.param_helper = param_helper
            self.multi_probe_pending = False
            self.z_min_position = lookup_minimum_z(config)
            self.results = []
            LookupZSteppers(config, self.mcu_probe.add_stepper)
            self.printer.lookup_object("pins").register_chip("probe", self)
            self.printer.register_event_handler(
                "homing:homing_move_begin", self._handle_homing_move_begin
            )
            self.printer.register_event_handler(
                "homing:homing_move_end", self._handle_homing_move_end
            )
            self.printer.register_event_handler(
                "homing:home_rails_begin", self._handle_home_rails_begin
            )
            self.printer.register_event_handler(
                "homing:home_rails_end", self._handle_home_rails_end
            )
            self.printer.register_event_handler(
                "gcode:command_error", self._handle_command_error
            )

        def _handle_homing_move_begin(self, hmove):
            if self.mcu_probe in hmove.get_mcu_endstops():
                self.mcu_probe.probe_prepare(hmove)

        def _handle_homing_move_end(self, hmove):
            if self.mcu_probe in hmove.get_mcu_endstops():
                self.mcu_probe.probe_finish(hmove)

        def _handle_home_rails_begin(self, homing_state, rails):
            endstops = [es for rail in rails for es, _ in rail.get_endstops()]
            if self.mcu_probe in endstops:
                self.mcu_probe.multi_probe_begin()
                self.multi_probe_pending = True

        def _handle_home_rails_end(self, homing_state, rails):
            endstops = [es for rail in rails for es, _ in rail.get_endstops()]
            if self.multi_probe_pending and self.mcu_probe in endstops:
                self.multi_probe_pending = False
                self.mcu_probe.multi_probe_end()

        def _handle_command_error(self):
            if self.multi_probe_pending:
                self.multi_probe_pending = False
                try:
                    self.mcu_probe.multi_probe_end()
                except Exception:
                    logging.exception("Homing multi-probe end")

        def setup_pin(self, pin_type, pin_params):
            if pin_type != "endstop" or pin_params["pin"] != "z_virtual_endstop":
                raise pins_mod.error(
                    "Probe virtual endstop only useful as endstop pin"
                )
            if pin_params["invert"] or pin_params["pullup"]:
                raise pins_mod.error(
                    "Can not pullup/invert probe virtual endstop"
                )
            return self.mcu_probe

        def start_probe_session(self, gcmd):
            self.mcu_probe.multi_probe_begin()
            self.results = []
            return self

        def run_probe(self, gcmd):
            toolhead = self.printer.lookup_object("toolhead")
            pos = toolhead.get_position()
            pos[2] = self.z_min_position
            speed = self.param_helper.get_probe_params(gcmd)["probe_speed"]
            phoming = self.printer.lookup_object("homing")
            self.results.append(phoming.probing_move(self.mcu_probe, pos, speed))

        def pull_probed_results(self):
            res = self.results
            self.results = []
            return res

        def end_probe_session(self):
            self.results = []
            self.mcu_probe.multi_probe_end()

    return HomingViaProbeHelper


def _probe_command_helper_factory(probe_mod, manual_probe_mod):
    run_single_probe = probe_mod.run_single_probe
    calc_probe_z_average = probe_mod.calc_probe_z_average

    class ProbeCommandHelper:
        def __init__(self, config, probe, query_endstop=None):
            self.printer = config.get_printer()
            self.probe = probe
            self.query_endstop = query_endstop
            self.name = config.get_name()
            gcode = self.printer.lookup_object("gcode")
            self.last_state = False
            gcode.register_command(
                "QUERY_PROBE", self.cmd_QUERY_PROBE, desc=self.cmd_QUERY_PROBE_help
            )
            self.last_z_result = 0.0
            gcode.register_command(
                "PROBE", self.cmd_PROBE, desc=self.cmd_PROBE_help
            )
            self.probe_calibrate_z = 0.0
            gcode.register_command(
                "PROBE_CALIBRATE",
                self.cmd_PROBE_CALIBRATE,
                desc=self.cmd_PROBE_CALIBRATE_help,
            )
            gcode.register_command(
                "PROBE_ACCURACY",
                self.cmd_PROBE_ACCURACY,
                desc=self.cmd_PROBE_ACCURACY_help,
            )
            gcode.register_command(
                "Z_OFFSET_APPLY_PROBE",
                self.cmd_Z_OFFSET_APPLY_PROBE,
                desc=self.cmd_Z_OFFSET_APPLY_PROBE_help,
            )

        def _move(self, coord, speed):
            self.printer.lookup_object("toolhead").manual_move(coord, speed)

        def get_status(self, eventtime):
            return {
                "name": self.name,
                "last_query": self.last_state,
                "last_z_result": self.last_z_result,
            }

        cmd_QUERY_PROBE_help = "Return the status of the z-probe"

        def cmd_QUERY_PROBE(self, gcmd):
            if self.query_endstop is None:
                raise gcmd.error("Probe does not support QUERY_PROBE")
            toolhead = self.printer.lookup_object("toolhead")
            print_time = toolhead.get_last_move_time()
            res = self.query_endstop(print_time)
            self.last_state = res
            gcmd.respond_info("probe: %s" % (["open", "TRIGGERED"][bool(res)],))

        cmd_PROBE_help = "Probe Z-height at current XY position"

        def cmd_PROBE(self, gcmd):
            pos = run_single_probe(self.probe, gcmd)
            gcmd.respond_info("Result is z=%.6f" % (pos[2],))
            self.last_z_result = pos[2]

        def probe_calibrate_finalize(self, kin_pos):
            if kin_pos is None:
                return
            z_offset = self.probe_calibrate_z - kin_pos[2]
            gcode = self.printer.lookup_object("gcode")
            gcode.respond_info(
                "%s: z_offset: %.3f\n"
                "The SAVE_CONFIG command will update the printer config file\n"
                "with the above and restart the printer." % (self.name, z_offset)
            )
            configfile = self.printer.lookup_object("configfile")
            configfile.set(self.name, "z_offset", "%.3f" % (z_offset,))

        cmd_PROBE_CALIBRATE_help = "Calibrate the probe's z_offset"

        def cmd_PROBE_CALIBRATE(self, gcmd):
            manual_probe_mod.verify_no_manual_probe(self.printer)
            params = self.probe.get_probe_params(gcmd)
            curpos = run_single_probe(self.probe, gcmd)
            self.probe_calibrate_z = curpos[2]
            curpos[2] += 5.0
            self._move(curpos, params["lift_speed"])
            x_offset, y_offset, _ = self.probe.get_offsets()
            curpos[0] += x_offset
            curpos[1] += y_offset
            self._move(curpos, params["probe_speed"])
            manual_probe_mod.ManualProbeHelper(
                self.printer, gcmd, self.probe_calibrate_finalize
            )

        cmd_PROBE_ACCURACY_help = "Probe Z-height accuracy at current XY position"

        def cmd_PROBE_ACCURACY(self, gcmd):
            params = self.probe.get_probe_params(gcmd)
            sample_count = gcmd.get_int("SAMPLES", 10, minval=1)
            toolhead = self.printer.lookup_object("toolhead")
            pos = toolhead.get_position()
            gcmd.respond_info(
                "PROBE_ACCURACY at X:%.3f Y:%.3f Z:%.3f"
                " (samples=%d retract=%.3f speed=%.1f lift_speed=%.1f)\n"
                % (
                    pos[0],
                    pos[1],
                    pos[2],
                    sample_count,
                    params["sample_retract_dist"],
                    params["probe_speed"],
                    params["lift_speed"],
                )
            )
            fo_params = dict(gcmd.get_command_parameters())
            fo_params["SAMPLES"] = "1"
            gcode = self.printer.lookup_object("gcode")
            fo_gcmd = gcode.create_gcode_command("", "", fo_params)
            probe_session = self.probe.start_probe_session(fo_gcmd)
            probe_num = 0
            while probe_num < sample_count:
                probe_session.run_probe(fo_gcmd)
                probe_num += 1
                pos = toolhead.get_position()
                liftpos = [None, None, pos[2] + params["sample_retract_dist"]]
                self._move(liftpos, params["lift_speed"])
            positions = probe_session.pull_probed_results()
            probe_session.end_probe_session()
            max_value = max(p[2] for p in positions)
            min_value = min(p[2] for p in positions)
            range_value = max_value - min_value
            avg_value = calc_probe_z_average(positions, "average")[2]
            median = calc_probe_z_average(positions, "median")[2]
            deviation_sum = sum((p[2] - avg_value) ** 2.0 for p in positions)
            sigma = (deviation_sum / len(positions)) ** 0.5
            gcmd.respond_info(
                "probe accuracy results: maximum %.6f, minimum %.6f, range %.6f, "
                "average %.6f, median %.6f, standard deviation %.6f"
                % (max_value, min_value, range_value, avg_value, median, sigma)
            )

        cmd_Z_OFFSET_APPLY_PROBE_help = "Adjust the probe's z_offset"

        def cmd_Z_OFFSET_APPLY_PROBE(self, gcmd):
            gcode_move = self.printer.lookup_object("gcode_move")
            offset = gcode_move.get_status()["homing_origin"].z
            if offset == 0:
                gcmd.respond_info("Nothing to do: Z Offset is 0")
                return
            z_offset = self.probe.get_offsets()[2]
            new_calibrate = z_offset - offset
            gcmd.respond_info(
                "%s: z_offset: %.3f\n"
                "The SAVE_CONFIG command will update the printer config file\n"
                "with the above and restart the printer."
                % (self.name, new_calibrate)
            )
            configfile = self.printer.lookup_object("configfile")
            configfile.set(self.name, "z_offset", "%.3f" % (new_calibrate,))

    return ProbeCommandHelper


def apply(probe_mod: Any):
    """Ensure probe helpers are present on ``probe_mod``.

    Returns a list of helper names that were added.
    """

    changed: List[str] = []

    manual_probe_mod = _try_import(
        "klippy.extras.manual_probe",
        "klipper.extras.manual_probe",
        "extras.manual_probe",
        "manual_probe",
    )
    if manual_probe_mod is None:
        _LOGGER.debug("manual_probe not available; skipping probe backports")
        return changed

    pins_mod = _try_import("klippy.pins", "klipper.pins", "pins")
    if pins_mod is None:
        _LOGGER.debug("pins module not available; skipping probe backports")
        return changed

    _ensure(probe_mod, "calc_probe_z_average", _calc_probe_z_average_factory, changed)
    _ensure(
        probe_mod,
        "lookup_minimum_z",
        lambda: _lookup_minimum_z_factory(manual_probe_mod),
        changed,
    )
    _ensure(probe_mod, "LookupZSteppers", _lookup_z_steppers_factory, changed)
    _ensure(probe_mod, "run_single_probe", lambda: _run_single_probe_factory(probe_mod), changed)
    _ensure(probe_mod, "ProbeOffsetsHelper", _probe_offsets_helper_factory, changed)
    _ensure(probe_mod, "ProbeParameterHelper", _probe_parameter_helper_factory, changed)
    _ensure(
        probe_mod,
        "ProbeSessionHelper",
        lambda: _probe_session_helper_factory(probe_mod),
        changed,
    )
    _ensure(
        probe_mod,
        "HomingViaProbeHelper",
        lambda: _homing_via_probe_helper_factory(probe_mod, pins_mod),
        changed,
    )
    _ensure(
        probe_mod,
        "ProbeCommandHelper",
        lambda: _probe_command_helper_factory(probe_mod, manual_probe_mod),
        changed,
    )

    return changed

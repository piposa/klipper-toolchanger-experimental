import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---- constants ----
POLLING_INTERVAL_DEFAULT = 1.0
DEFAULT_WATT             = 60.0

AT_HEAT_TOLERANCE       = 2     # °C below target still considered heating
WARMUP_GUARD            = 5     # s after a target change before considering rescue
RESCUE_URGENCY_FLOOR    = 0.10  # ignore tiny urgency below this

VERBOSE = False

# Built‑in derating models (factor vs temperature °C). Coarse approximations.
# whats v1? v3? no clue. i stole this from reddit.
BUILTIN_MODELS = {
    'none':        [(0,1.0),(400,1.0)],
    'bambulab':    [(25,1.00),(100,0.85),(200,0.70),(260,0.62),(320,0.52)],
    'v1':          [(25,1.00),(100,0.82),(200,0.66),(260,0.58),(320,0.48)],
    'rapido_v2':   [(25,1.00),(100,0.90),(200,0.78),(260,0.70),(320,0.63)],
    'v3':          [(25,1.00),(100,0.88),(200,0.76),(260,0.68),(320,0.60)],
    'triangle70':  [(25,1.00),(100,0.80),(200,0.62),(260,0.54),(320,0.45)],
}


def _clamp(val, lo, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


@dataclass
class HeaterSnapshot:
    temp: float
    target: float
    slope: Optional[float]
    base_w: float
    budget_cap_w: float
    ctrl_cap: float
    need_heat: bool
    last_target_change: float
    is_watt_mode: bool
    orig_max_power: float


@dataclass
class GroupSnapshot:
    active_names: List[str]
    snapshots: Dict[str, HeaterSnapshot]
    total_budget_cap_w: float


@dataclass
class WeightPlan:
    weights: Dict[str, float]
    urgent: List[str]
    triggered: bool
    details: List[tuple]


@dataclass
class HeaterRec:
    """Per-heater runtime record"""
    heater: object
    rated_watt: float
    vh: object
    vh_params: tuple
    sync_mpc_power: bool = False
    control_id: Optional[int] = None
    control_type: Optional[str] = None
    is_watt_mode: bool = False
    orig_max_power: float = 1.0
    prev_temp: Optional[float] = None
    prev_time: Optional[float] = None
    prev_target: Optional[float] = None
    last_target_change: float = 0.0
    slope_ema: Optional[float] = None

    def update_control_info(self, control):
        self.control_id = id(control)
        get_type = getattr(control, 'get_type', None)
        try:
            control_type = get_type() if callable(get_type) else None
        except Exception:
            control_type = None
        self.control_type = control_type
        self.is_watt_mode = control_type == 'mpc'
        default = self.rated_watt if self.is_watt_mode else 1.0
        value = getattr(control, 'heater_max_power', default)
        try:
            value = float(value)
        except Exception:
            value = default
        if self.is_watt_mode:
            if value <= 0.0:
                value = self.rated_watt
            if self.sync_mpc_power:
                try:
                    if hasattr(control, 'const_heater_power'):
                        control.const_heater_power = value
                    control.heater_max_power = value
                except Exception:
                    pass
        elif value < 0.0:
            value = 0.0
        self.orig_max_power = value

class HeaterPowerDistributor:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.name = config.get_name().split()[-1]

        self.heater_list           = config.getlist('heaters')
        self.max_total_watts       = config.getfloat('max_power',                                above=0.0)
        self.poll_interval         = config.getfloat('poll_interval',  POLLING_INTERVAL_DEFAULT, above=0.0)
        self.active_extruder_boost = config.getfloat('active_extruder_boost', 1.0,               above=0.0)
        self.safe_distribution     = config.getboolean('safe_distribution', True)
        self.sync_mpc_power        = config.getboolean('sync_mpc_power', False)

        model_name = config.getchoice('model', list(BUILTIN_MODELS) + ['custom'], default='none')
        powers     = config.getfloatlist('powers', (DEFAULT_WATT,))

        if len(powers) == 1:
            self.rated_watts_cfg = [float(powers[0])] * len(self.heater_list)
        elif len(powers) == len(self.heater_list):
            self.rated_watts_cfg = [float(p) for p in powers]
        else:
            raise config.error("'powers' length (%d) must be 1 or match heaters count (%d)" % (len(powers), len(self.heater_list)))

        if model_name == 'custom':
            pairs  = config.getlists('model_points', seps=(',', '\n'), count=2, parser=float)
            points = [(t, f) for (t, f) in pairs]
            if len(points) < 2:
                raise config.error("'model_points' must have at least two 'temp, factor' pairs")
        else:
            points = BUILTIN_MODELS[model_name]
        self.model_curve = _LinearCurve(points)
        self.model_name  = model_name

        # Runtime state
        self.heaters = {}     # name -> HeaterRec
        self.last_time = None
        self.rescue_on = False
        self.last_notify_time = 0.0

        self.pheaters = self.printer.load_object(config, 'heaters')
        self.gcode = self.printer.lookup_object('gcode')
        self.gcode.register_mux_command(
            'SET_HEATER_DISTRIBUTOR', 'GROUP', self.name,
            self.cmd_SET_HEATER_DISTRIBUTOR,
            desc=self.cmd_SET_HEATER_DISTRIBUTOR_help
        )
        self.printer.register_event_handler('klippy:ready', self._handle_ready)

    # ---------------------------- setup ----------------------------
    def _handle_ready(self):
        for idx, name in enumerate(self.heater_list):
            heater = self.pheaters.lookup_heater(name)
            rated = float(self.rated_watts_cfg[idx])

            # verify_heater params (best-effort)
            vh = self.printer.lookup_object('verify_heater %s' % (name,), None)
            if vh is not None:
                hg   = float(getattr(vh, 'heating_gain', 2.0))
                cgt  = float(getattr(vh, 'check_gain_time', 60.0 if name == 'heater_bed' else 20.0))
                hyst = float(getattr(vh, 'hysteresis', 5.0))
            else:
                cgt_default = 60.0 if name == 'heater_bed' else 20.0
                hg, cgt, hyst = 2.0, cgt_default, 5.0

            self.heaters[name] = HeaterRec(
                heater=heater,
                rated_watt=rated,
                vh=vh,
                vh_params=(hg, cgt, hyst),
                sync_mpc_power=self.sync_mpc_power,
            )
            if self.safe_distribution:
                try:
                    heater.control.heater_max_power = 0.0
                except Exception:
                    pass

        reactor = self.printer.get_reactor()
        reactor.register_timer(self._update_callback, reactor.monotonic() + self.poll_interval)

        logging.info(
            "HeaterPowerDistributor '%s' heaters=%s rated=%sW budget=%.1fW model=%s",
            self.name,
            list(self.heaters.keys()),
            [self.heaters[h].rated_watt for h in self.heater_list],
            self.max_total_watts,
            self.model_name,
        )

    # ---------------------------- commands ----------------------------
    cmd_SET_HEATER_DISTRIBUTOR_help = "Set max total power (watts) for this group"
    def cmd_SET_HEATER_DISTRIBUTOR(self, gcmd):
        power = gcmd.get_float('POWER', above=0.0)
        self.max_total_watts = power
        gcmd.respond_info("%s: budget set to %.1f W." % (self.name, self.max_total_watts))

    # ---------------------------- helpers ----------------------------
    def _derate_factor(self, temp_c):
        try:
            f = float(self.model_curve.at(temp_c))
        except Exception:
            f = 1.0
        return max(0.0, min(1.0, f))

    def _caps_for(self, rec, base_w):
        """Return (budget_cap_w, ctrl_cap) for this heater at current temp."""
        if rec.is_watt_mode:
            budget_cap = max(0.0, min(base_w, rec.orig_max_power))
            ctrl_cap = budget_cap
            if self.sync_mpc_power:
                # Drive MPC with the derated available watts so its duty math tracks reality.
                try:
                    ctrl = rec.heater.control
                    if hasattr(ctrl, 'const_heater_power'):
                        ctrl.const_heater_power = budget_cap
                    ctrl.heater_max_power = budget_cap
                    # If control tracks last_power, clamp it to avoid brief >100% readings.
                    if hasattr(ctrl, 'last_power') and getattr(ctrl, 'last_power') > budget_cap:
                        ctrl.last_power = budget_cap
                except Exception:
                    pass
            return budget_cap, ctrl_cap
        budget_cap = max(0.0, base_w * rec.orig_max_power)
        ctrl_cap = max(0.0, rec.orig_max_power)
        return budget_cap, ctrl_cap

    def _calc_slope(self, rec, temp, eventtime, tau=1.5):
        """EMA slope calculator in °C/s; returns (slope or None)."""
        if rec.prev_time is None or rec.prev_temp is None:
            rec.prev_temp = temp
            rec.prev_time = eventtime
            return None
        dt = max(1e-6, eventtime - rec.prev_time)
        inst = (temp - rec.prev_temp) / dt
        alpha = dt / (tau + dt)
        prev_ema = rec.slope_ema
        ema = inst if prev_ema is None else (alpha * inst + (1.0 - alpha) * prev_ema)
        rec.slope_ema = ema
        rec.prev_temp = temp
        rec.prev_time = eventtime
        return ema

    def _snapshot(self, eventtime):
        """Sample all heaters, build a GroupSnapshot."""
        snapshots = {}
        active = []
        for name, rec in self.heaters.items():
            heater = rec.heater
            control = heater.control
            if rec.control_id != id(control):
                rec.update_control_info(control)
            temp, target = heater.get_temp(eventtime)
            derate = self._derate_factor(temp)
            base_w = max(0.0, rec.rated_watt * derate)
            budget_cap_w, ctrl_cap = self._caps_for(rec, base_w)

            if rec.prev_target is None or rec.prev_target != target:
                rec.last_target_change = eventtime
                rec.prev_target = target

            slope = self._calc_slope(rec, temp, eventtime)

            need_heat = (target > 0.0) and (temp <= (target - AT_HEAT_TOLERANCE)) and budget_cap_w > 0.0

            snapshots[name] = HeaterSnapshot(
                temp=temp,
                target=target,
                slope=slope,
                base_w=base_w,
                budget_cap_w=budget_cap_w,
                ctrl_cap=ctrl_cap,
                need_heat=need_heat,
                last_target_change=rec.last_target_change,
                is_watt_mode=rec.is_watt_mode,
                orig_max_power=rec.orig_max_power,
            )
            if need_heat:
                active.append(name)

        total_cap_w = sum(snapshots[n].budget_cap_w for n in active)
        return GroupSnapshot(active_names=active, snapshots=snapshots, total_budget_cap_w=total_cap_w)

    def _detect_urgent(self, snapshot, budget_w, eventtime):
        """Return (urgent_list, details, triggered_flag)."""
        urgent = []
        details = []
        power_limited = budget_w < snapshot.total_budget_cap_w - 1e-9
        if not power_limited:
            return urgent, details, False

        seen = set()
        for name in snapshot.active_names:
            if name in seen:
                continue
            seen.add(name)
            snap = snapshot.snapshots[name]
            rec = self.heaters[name]
            vh = rec.vh
            hg, cgt, vh_hyst = rec.vh_params
            base_slope = hg / cgt

            # warmup guard / near target guard
            if (eventtime - snap.last_target_change) < WARMUP_GUARD:
                continue
            if snap.temp >= snap.target - vh_hyst:
                continue

            urgency = 0.0
            if vh is not None and getattr(vh, 'approaching_target', False):
                goal_temp = getattr(vh, 'goal_temp', snap.temp)
                goal_time = getattr(vh, 'goal_systime', eventtime)
                delta = max(0.0, goal_temp - snap.temp)
                t_rem = max(0.5, goal_time - eventtime)
                req_slope = delta / t_rem
                if base_slope > 1e-9:
                    urgency = max(urgency, req_slope / base_slope - 1.0)

            if snap.slope is not None and base_slope > 1e-9:
                urgency = max(urgency, (base_slope - snap.slope) / base_slope)

            if urgency > RESCUE_URGENCY_FLOOR:
                urgent.append(name)
                scale_hint = 1.0 + urgency * max(1, len(snapshot.active_names))
                details.append((name, round(urgency, 3), round(scale_hint, 2)))
        triggered = bool(urgent)
        return urgent, details, triggered

    def _plan_weights(self, snapshot, active_extruder_name, budget_w, eventtime):
        weights = {n: 1.0 for n in snapshot.active_names}
        boost = float(self.active_extruder_boost)
        if boost != 1.0 and active_extruder_name in weights:
            weights[active_extruder_name] = boost

        urgent, details, triggered = self._detect_urgent(snapshot, budget_w, eventtime)
        return WeightPlan(weights=weights, urgent=urgent, triggered=triggered, details=details)

    def _get_active_extruder_name(self, eventtime):
        th = self.printer.lookup_object('toolhead', None)
        if th is None:
            return None
        try:
            status = th.get_status(eventtime) or {}
            return status.get('extruder')
        except Exception:
            return None

    # ---------------------------- planning ----------------------------

    def _water_fill(self, active_names, weights, eff_cap_w, target_budget):
        alloc_w   = {n: 0.0 for n in active_names}
        remaining = target_budget
        unsat     = set(active_names)
        while unsat and remaining > 1e-9:
            total_w = sum(weights[n] for n in unsat)
            if total_w <= 1e-12:
                break
            share_per_weight = remaining / total_w
            newly_sat = set()
            for n in list(unsat):
                room = eff_cap_w[n] - alloc_w[n]
                want = weights[n] * share_per_weight
                add  = want if want <= room else room
                if add > 0.0:
                    alloc_w[n] += add
                if (eff_cap_w[n] - alloc_w[n]) <= 1e-12:
                    newly_sat.add(n)
            remaining = target_budget - sum(alloc_w.values())
            if not newly_sat:
                break
            unsat -= newly_sat
        return alloc_w

    def _apply_caps(self, alloc_w, snapshot):
        for name, rec in self.heaters.items():
            control = rec.heater.control
            if rec.control_id != id(control):
                rec.update_control_info(control)
            snap = snapshot.snapshots.get(name)
            if name in alloc_w and snap is not None:
                budget_cap = snap.budget_cap_w
                if budget_cap <= 1e-9:
                    permitted = 0.0
                    if rec.is_watt_mode and self.sync_mpc_power:
                        try:
                            if hasattr(control, 'const_heater_power'):
                                control.const_heater_power = rec.orig_max_power
                            control.heater_max_power = rec.orig_max_power
                            if hasattr(control, 'last_power') and getattr(control, 'last_power') > rec.orig_max_power:
                                control.last_power = rec.orig_max_power
                        except Exception:
                            pass
                else:
                    share = _clamp(alloc_w[name] / budget_cap, 0.0, 1.0)
                    permitted = share * snap.ctrl_cap
                    if not rec.is_watt_mode:
                        permitted = max(0.0, min(rec.orig_max_power, permitted))
                control.heater_max_power = permitted
            else:
                control.heater_max_power = rec.orig_max_power
                if rec.is_watt_mode and self.sync_mpc_power:
                    try:
                        if hasattr(control, 'const_heater_power'):
                            control.const_heater_power = rec.orig_max_power
                        if hasattr(control, 'last_power') and getattr(control, 'last_power') > rec.orig_max_power:
                            control.last_power = rec.orig_max_power
                    except Exception:
                        pass

    # ---------------------------- loop ----------------------------
    def _update_callback(self, eventtime):
        if self.last_time is None:
            self.last_time = eventtime
            return eventtime + self.poll_interval
        self.last_time = eventtime

        active_extruder = self._get_active_extruder_name(eventtime)
        snapshot = self._snapshot(eventtime)

        # If nobody needs heat or capacities collapsed, restore and bail
        if not snapshot.active_names or snapshot.total_budget_cap_w <= 1e-9:
            for name, rec in self.heaters.items():
                control = rec.heater.control
                if rec.control_id != id(control):
                    rec.update_control_info(control)
                snap = snapshot.snapshots.get(name)
                if snap is not None and snap.target > 0.0:
                    control.heater_max_power = snap.budget_cap_w
                    if rec.is_watt_mode and self.sync_mpc_power:
                        try:
                            if hasattr(control, 'const_heater_power'):
                                control.const_heater_power = snap.budget_cap_w
                            if hasattr(control, 'last_power') and getattr(control, 'last_power') > snap.budget_cap_w:
                                control.last_power = snap.budget_cap_w
                        except Exception:
                            pass
                elif self.safe_distribution:
                    control.heater_max_power = 0.0
                else:
                    control.heater_max_power = rec.orig_max_power
            return eventtime + self.poll_interval

        budget_w = max(0.0, float(self.max_total_watts))
        target_budget = min(budget_w, snapshot.total_budget_cap_w)

        plan = self._plan_weights(snapshot, active_extruder, budget_w, eventtime)

        if plan.triggered and (not self.rescue_on or (eventtime - self.last_notify_time) > 5.0):
            self.rescue_on = True
            self.last_notify_time = eventtime
            msg = ("%s: verify-heater assist active; power-limited. urgent=%s boosted: %s" % (
                self.name,
                list(plan.urgent)[:4],
                ", ".join(["%s(urg=%s,x%s)" % (n,u,sc) for (n,u,sc) in plan.details[:4]])
            ))
            if VERBOSE:
                self.gcode.respond_info(msg)
            else:
                logging.info(msg)

        elif not plan.triggered:
            self.rescue_on = False

        caps = {n: snapshot.snapshots[n].budget_cap_w for n in snapshot.active_names}
        alloc_w = self._alloc_priority(list(plan.urgent), list(snapshot.active_names), plan.weights, caps, target_budget)
        self._apply_caps(alloc_w, snapshot)
        return eventtime + self.poll_interval

    # ---------------------------- priority allocator ----------------------------
    def _alloc_priority(self, urgent_names, active_names, weights, eff_cap_w, target_budget):
        # Nothing to do
        if target_budget <= 1e-9:
            return {n: 0.0 for n in active_names}
        urgent_set = set(urgent_names)
        urgent_cap = sum(eff_cap_w[n] for n in urgent_set)
        alloc = {n: 0.0 for n in active_names}

        if not urgent_set:
            return self._water_fill(active_names, weights, eff_cap_w, target_budget)

        if urgent_cap <= target_budget + 1e-9:
            # Give urgent heaters their full instantaneous caps first
            for n in urgent_set:
                alloc[n] = eff_cap_w[n]
            remaining = target_budget - urgent_cap
            others = [n for n in active_names if n not in urgent_set]
            if remaining > 1e-9 and others:
                fill = self._water_fill(others, {k: weights[k] for k in others}, eff_cap_w, remaining)
                alloc.update(fill)
            return alloc

        # Not enough budget even for urgent set: share only among urgent
        fill = self._water_fill(list(urgent_set), {k: 1.0 for k in urgent_set}, eff_cap_w, target_budget)
        alloc.update(fill)
        return alloc


class _LinearCurve:
    def __init__(self, points):
        pts = sorted([(float(x), float(y)) for x, y in points], key=lambda p: p[0])
        self.x = [p[0] for p in pts]
        self.y = [p[1] for p in pts]
    def at(self, x):
        if x <= self.x[0]:
            return self.y[0]
        if x >= self.x[-1]:
            return self.y[-1]
        lo, hi = 0, len(self.x)-1
        while hi - lo > 1:
            mid = (lo+hi)//2
            if x < self.x[mid]:
                hi = mid
            else:
                lo = mid
        x0, y0 = self.x[lo], self.y[lo]
        x1, y1 = self.x[hi], self.y[hi]
        t = (x - x0) / (x1 - x0)
        return y0 + t*(y1 - y0)


def load_config_prefix(config):
    return HeaterPowerDistributor(config)

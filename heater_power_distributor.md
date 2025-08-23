# Heater Power Distributor

Bound your printer’s total heater power and share it intelligently between multiple heaters (e.g. several hotends). 
This module caps **per‑tick heater power** so you can stay under a PSU limit while still getting fast warmups, 
optionally prioritising the active extruder or any *urgent* heaters *(as indicated by `verify_heater` becoming uneasy with what we are doing)*.

> Badly chosen power budgets can trip `verify_heater` or cause slow heating.
> *(If 5 toolheads need 70W to stay hot, they need 70W to stay hot, setting a group limit for those 5 below 70W will still trigger `verify_heater`.)*

---

## Why?
Multi‑tool printers can exceed PSU capacity if several heaters ramp at once.
This extra sets a **global wattage budget** for a group of heaters and dynamically distributes that budget. 
It respects each heater’s own `heater_max_power`.

---

## Features
- Global **max power budget** (in watts) per heater group.
- Per‑heater **wattage** and **temperature derating model** *(built‑ins for common hotends, or a custom curve)*.
- **Active extruder boost** to give the printing tool a bigger share.
- **Rescue mode** that prioritises heaters likely to trip `verify_heater` based on their approach rate to target.
- just in case - a gcode command to change the budget on the fly.
- 'detatches' 2°C below setpoint to not mess with PID

> [!WARNING]
> Make sure youre still using a sane power rated PSU, if 5 tools turn on at the same time,
> it will take between `0` and `poll_interval` seconds before the power gets reduced.
> most PSUs dont trip this fast and are rated for ~2x current for atleast 10 seconds. a very fast electronic fuse however may still trigger.

---

## Configuration

Create a **group** defining which heaters share a power budget.

```hcl
# Example: 3 hotends sharing 150 W, with Rapido V2 derating
[heater_power_distributor hotends]
heaters: extruder, extruder1, extruder2
max_power: 150                # total budget for the group (watts)
powers: 60, 60, 60            # rated W per heater, defaults to 60
model: rapido_v2              # none|bambulab|v1|rapido_v2|v3|triangle70|custom, defaults to none
active_extruder_boost: 1.8    # multiply weight of active tool (1.0 = equal-share), defaults to 1
poll_interval: 1.0            # seconds between updates, defaults to 1
```

If you prefer your **own derating curve**, set `model: custom` and provide `temperature, factor pairs` (`factor` is a 0–1 multiplier on rated watts):

```hcl
[heater_power_distributor hotends]
# ...
model: custom
model_points:
  25, 1.00
  100, 0.85
  200, 0.70
  260, 0.62
  320, 0.52
```

since they are named, multiple sections are supported.
so you may also make one for hotends and one for (for example) bed + chamber (if your bed draws 1kW, and chamber heater 500W, but your outlet only does 1kW, you can similarly clamp it).

> [!TIP]
> - If you set `powers` to a single number, it’s applied to all heaters in the group.
> - The module honours each heater’s own `heater_max_power` limit; consider reverting those back to 1.0 if they were changed and let the distributor handle sharing.
> - Built‑in derating models are coarse approximations; use `custom` if you have measurements.

---

## G‑code

Change the group’s power budget at runtime:

```hcl
# Set budget to 200 W for the [heater_power_distributor hotends] group
SET_HEATER_DISTRIBUTOR GROUP=hotends POWER=200
```




---

## How it works (nutshell)

- Each tick the module samples all group heaters (temperature, target), computes a **temperature‑derated watt cap** from each heater’s rated watts, then distributes the **global budget** via a weighted “water‑fill” algorithm.
- When the budget is tight, a **rescue step** detects heaters at risk of failing `verify_heater` (based on the hotend’s expected heat‑up slope from its `verify_heater` gains, plus the observed temperature slope) and **prioritises** them until they catch up.
- If power isn’t limited (your budget ≥ sum of instantaneous caps), everyone just gets their own cap and the module leaves them alone.
- When nobody needs heat, all heaters’ `heater_max_power` are restored to their original values.

---



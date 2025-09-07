# heater_chamber_fan

A small extra to control a filter which is also used to heat your chamber.  
like `[temperature_fan]` but for heating and more unhinged.

## Features

* Heater‑style target setting
* supporting fan semantics: `max_power`, `off_below` and the new  `min_power`
* post‑print linger with configurable duration and and power
* Manual override via `SET_FAN_SPEED` *(in case its useful/needed)*

## Requirements

* A chamber temperature sensor which gets configured in this section
* A controllable fan output (`fan_pin`)

### Minimal example

```ini
[heater_chamber_fan chamber]
# Sensor (same options as other heaters)
sensor_type: Generic 3950
sensor_pin: PA0
pullup: True

fan_pin: PB0

min_temp: 0                         # °C (>= -273.15)
max_temp: 80                        # °C (> min_temp)
control: pid                        # pid | watermark
smooth_time: 2.0                    # s, temp smoothing before control

min_power: 0.15                     # lower bound while controlling
max_power: 0.80                     # upper bound

gate_against: heater_bed
gate_mode: target                   # target | current | either | None

auto_linger: True                   # start linger when print finishes
linger_time: 120                    # seconds
linger_power: 0.20                  # duty during linger
```

## G‑code

`SET_HEATER_TEMPERATURE HEATER=<name> TARGET=<temp> [WAIT=0|1]`

Set the chamber target. With `WAIT=1`, If gating is enabled and the gating condition isn’t met, it raises a gcmd error  
> (mainly to prevent you from accidentally setting a wait condition it cannot meet)  

`SET_FAN_SPEED` stays regular.  
  
`CHAMBER_FAN_LINGER HEATER=<name> [SECONDS=<s>] [POWER=<0..1>]`  
  
Start or restart a linger period that runs at a fixed `POWER` for `SECONDS`, then stops.
Also used internally when a print ends and `auto_linger: True`.

example that runs the fan for a further hour at 25% before turning off.
```gcode
CHAMBER_FAN_LINGER HEATER=chamber SECONDS=3600 POWER=0.25
```

## Behavior notes

* **Manual override**: `SET_FAN_SPEED` switches to manual mode. The control loop remains idle until you call `SET_HEATER_TEMPERATURE` again.
* **Gating**: With `gate_against` enabled, `WAIT=1` returns immediately unless the other heater is warm enough based on `gate_mode`:

  * `target`: other heater’s **target** ≥ your chamber target
  * `current`: other heater’s **current** ≥ your chamber target
  * `either`: either condition is sufficient

## info in get_status()
```ini
{
  "temperature": <float>,
  "target": <float>,
  "power": <float>,
  "state": "idle|active|linger",
  "linger_time_left": <float>
}
```

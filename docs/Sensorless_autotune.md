
# Klipper Sensorless Autotune

A Klipper extra for automatic tuning of TMC drivers’ **StallGuard thresholds** (SGT/SGTHRS).

## Features
- Automated sweep of driver sensitivity values.
- Detects minimum working sensitivity and safe window.
- Works with TMC2209, TMC5160, and similar drivers using StallGuard.
- calls your regular G28 

## Setup

add to `printer.cfg`
```ini
[sensorless_auto_tune]
speed: 100 # (your restore speed)
```

Then run in console with:
```
SENSORLESS_AUTOTUNE AXIS=X
SENSORLESS_AUTOTUNE AXIS=Y
```

> Optional parameters:
> 
> * `MIN_MOVE` → minimum distance to count as an actual endstop trigger (default=`auto`).
> * `WINDOW` → safety margin as 'overtravel' (default: `0.60`).
> * `START` / `STOP` → sweep bounds (driver raw values). if omitted starts at highest sens.

## Example
<img src="../media/sensorless_auto_tune-ezgif.com-optimize.gif" width="600">

## Requirements
* TMC driver with sensorless homing support (SGT/SGTHRS).
* Correct `virtual_endstop` configured for each axis.
* correctly setup sensorless

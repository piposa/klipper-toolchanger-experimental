# Klipper Sensorless Autotune

A Klipper extra for automatic tuning of TMC drivers’ **StallGuard thresholds** (SGT/SGTHRS) and finding your exact axis extremes.

## Features

* Automated sweep of driver sensitivity values.
* Detects minimum working sensitivity and safe window.
* Works with TMC2209, TMC5160, and similar drivers using StallGuard.
* Extra command to automatically infer **axis min/max travel** by homing both directions.

## Setup

Add to `printer.cfg`:

```ini
[sensorless_auto_tune]
restore_speed: 100  # optional, mm/s; if omitted falls back to the axis' homing speed
```

Run in console:

```
SENSORLESS_AUTOTUNE AXIS=X
SENSORLESS_AUTOTUNE AXIS=Y
```

### Optional parameters

* `MIN_MOVE` → minimum travel required to count as a valid endstop trigger (default = auto-detected).
* `WINDOW` → tolerance margin for “too late” triggers, mm (default = `0.60`).
* `START` / `STOP` → raw driver register sweep bounds; if omitted, starts at the most sensitive end.

---

## Finding axis limits

```
FIND_AXIS_CONSTRAINTS AXIS=X
```

* Homes to the **opposite** end first, then back to the normal end.
* Sums both travel distances to calculate the **total usable span**.
* Prints and stages new `position_min` / `position_max` for the axis.
  Example:

  ```
  [X] travel=350.000 mm (|opposite|=140.000 + |normal|=210.000); endstop@=350.000 ⇒ min=0.000, max=350.000
  Staged for SAVE_CONFIG: [stepper_x] position_min=0.000, position_max=350.000
  ```

> ⚠️ Note: `SAVE_CONFIG` cannot overwrite values defined in an included file.
> For `position_min/max` to be staged cleanly, ensure they are **not** defined in an include.

---

## Example

<img src="../media/sensorless_auto_tune-ezgif.com-optimize.gif" width="600">

---

## Requirements

* TMC driver with StallGuard (SGT/SGTHRS).
* `virtual_endstop` correctly set up for each axis.
* Sensorless homing already working manually.
* your wanted Homing endstop positions (`position_endstop`) and roughly correct opposite bound (pos_min/pos_max) so the sweep knows which way to move.

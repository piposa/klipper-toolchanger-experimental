# klipper-toolchanger

Toolchanging extension for Klipper.
Provides the basic structure and links into GCodes.  

Not opinionated how the tool change happens and suitable for physical tools as well as MMUs. 
All the actual tool changing motions to be provided as gcode macros.

# Status

 * Single toolchanger works well. 

TODO: 

 * Cascading tools support.
 * Save current tool on restart.

# Lifecycle

This is the overall lifecycle of the toolchanger.

The initialization can be done either manually from PRINT_START or automatically
on home or first toolchange.

![Lifecycle](../media/lifecycle.png)

The exact sequence how each step is executed:
![Sequence](../media/Sequence.png)

---

## Config

### [toolchanger]

Configures common tool changing parameters. 
More than one toolchanger can be configured, with arbitrary names.
~~The unnamed toolchanger is always considered the main one and others can be 
connected to a tool in the main toolchanger.~~(**TODO**)  

Any parameter that can be set on a tool, can be set on the toolchanger as well
and will provide a default value for all of its tools.

```
initialize_on: <manual|home|first-use>   default: first-use)
initialize_gcode: <template>            
verify_tool_pickup: <True|False>         default: True)
require_tool_present: <True|False>       default: False)
uses_axis: <xyz>                         default: xyz)
on_axis_not_homed: <abort|home>          default: abort)
perform_restore_move: <True|False>       default: True)
pickup_gcode: <template> 
dropoff_gcode: <template> 
before_change_gcode: <template>         
after_change_gcode: <template>          
error_gcode: <template>                  required for VERIFY_TOOL_DETECTED ASYNC=1
recover_gcode: <template>                Template used to recover after error_gcode (see INITIALIZE_TOOLCHANGER RECOVER=1).
on_tool_mounted_gcode: <template>        These templates fire on detection state changes but not during toolchanges or probing.
on_tool_removed_gcode: <template>       
transfer_fan_speed: <True|False>         default: True)
params_*: <python literal>               (exposed in templates and inherited by tools)
(and all the other ones that tools can have)
```
### [tool]

Defines a tool that can be selected.
Normally a tool has an assigned extruder, fans and associated printer config,
like pressure advance. But can be purely virtual, like slot in an MMU unit.
See [command reference](#Gcodes) for how to control tools.
Tool params, gcode offsets, restore axis.... all inherit from `[toolchanger]` and can override.  
```
extruder: <extruder>                    
extruder_stepper: <extruder_stepper>     optional extra stepper to sync via SYNC_EXTRUDER_MOTION
fan: <fan_generic>                       
detection_pin: <pin>                     required if any tool has detection, enables on_tool_mounted_gcode and on_tool_removed_gcode
tool_number: <n>=0..                     registers T<n>
gcode_x_offset: <mm>                     
gcode_y_offset: <mm>                     
gcode_z_offset: <mm>
# most if not all the templates from toolchanger are equally able to overwrite the "default" ones inherited from toolchanger.               
params_*: <python literal>               overrides inherited parameters from toolchanger
```

## templates
### avalible in all templates:
* tool - currently active tool or None if no tool. This is the actual object, not a name, so can use directly, ie `tool.fan`.
* toolchanger - same as printer.toolchanger.

### avalible in before_change_gcode...pickup_gcode etc have the following:
* `dropoff_tool` - name of the tool that is being dopped off, or None if not a dropoff operation.
* `pickup_tool` - name of the tool that is being picked up, or None if not a dropoff operation.
### pickup_gcode and dropoff_gcode specific
* `restore_position.X .Y .Z` - coordinates to restore the position to after toolchange. 
   Only the ones requested are set.  
   Adjusted to account for tool offsets.
* `start_position.X .Y .Z` - coordinates before toolchange, regardless of what is requested. 
   Adjusted to account for tool offsets. 
### on_tool_mounted_gcode and on_tool_removed_gcode specific
* `detected_tool` tool object of just mounted tool (or none if not a mount)
* `removed_tool` tool object of just removed tool (or none if not a remove)

---

## Gcodes

### [toolchanger]

The following commands are available when toolchanger is loaded.

### INITIALIZE_TOOLCHANGER
`INITIALIZE_TOOLCHANGER [TOOLCHANGER=toolchanger] [TOOL_NAME=<name>] [T=<number>] [RECOVER=0]`: 
Initializes or Re-initializes the toolchanger state. Sets toolchanger status to `ready`.

The default behavior is to auto-initialize on first tool selection call.
Always needs to be manually re-initialized after a `SELECT_TOOL_ERROR`. 
If `TOOL_NAME` is specified, sets the active tool without performing any tool change
gcode. The after_change_gcode is always called. `TOOL_NAME` with empty name unselects
tool.

Experimental: If `RECOVER=1` is specified, `recover_gcode` is run and toolehad is moved to restore_axis position. 

### ASSIGN_TOOL
`ASSIGN_TOOL TOOL=<name> N=<number>`: Assign tool to specific tool number.
Overrides any assignments set up by `tool.tool_number`.
Sets up a corresponding T<n> and M106/M107 T<index> commands.
Does *not* change the active tool.

### SELECT_TOOL
`SELECT_TOOL TOOL=<name> [RESTORE_AXIS=xyz]`: Select the active tool.
The toolhead will be moved to the previous position on any axis specified in
`RESTORE_AXIS` value. Slicer Gcode normally use `T0`, `T1`, `T2`,... to select a tool.
Printer config should contain macros to map them to corresponding tool names,
or set `tool.tool_number:` to auto register a T macro.

The selection sequence is as follows:

- gcode state is saved
- toolchanger.before_change_gcode is run
- current extruder and fan are deactivated, if changed
- current_tool.dropoff_gcode is run - if a tool is currently selected
- new_tool.pickup_gcode is run
- new extruder and fan are activated, if changed
- toolchanger.after_change_gcode is run
- gcode state is restored, without move
- new tool is moved to the gcode position, according to RESTORE_AXIS and force restore.

~~If the tools have parents, their corresponding dropoff/pickup gcode is also run.~~(TODO)  

### SELECT_TOOL_ERROR
`SELECT_TOOL_ERROR [MESSAGE=]`: Signals failure to select a tool. 
Can be called from within tool macros during SELECT_TOOL and will abort any
remaining tool change steps and put the toolchanger starting the selection in
`ERROR` state. Then runs `error_gcode` if one is provided, 

### UNSELECT_TOOL
`UNSELECT_TOOL [RESTORE_AXIS=]`: Unselect active tool without selecting a new one.

Performs only the first part of select tool, leaving the printer with no tool 
selected.

### VERIFY_TOOL_DETECTED
`VERIFY_TOOL_DETECTED [TOOL=<name>] [T=<number>] [ASYNC=0] [WINDOW=0.005]`: Check if detected tool 
matches the expected tool. runs error gcode or shuts down klipper.
Does nothing if tool detection pin is not configured.
* If ASYNC=0, will continously check if the tool has been detected, and time out when all queued moves are complete. *(for fastest detection, run as late as you can allow safely)*  
* If ASYNC=1, will return immediately and perform the check in background, call this at the point where you know you `**just** triggered the detection pin for best results. On non detect this may however mean that the toolchange and restore move long finished before the error_gcode commands hit the que! beware of that.
If youre using async and get false triggers, raise WINDOW, otherwise keep it lower for faster fault detection.  

A verification failure will:
 - abort in-progress toolchange.
 - put the toolchanger in `ERROR` state and run `error_gcode` if one is provided.

### SET_TOOL_TEMPERATURE
`SET_TOOL_TEMPERATURE [TOOL=<name>] [T=<number>]  TARGET=<temp> [WAIT=0]`: Set tool temperature.

### TEST_TOOL_DOCKING
`TEST_TOOL_DOCKING`: Dock and undock current tool.

### SET_TOOL_PARAMETER
`SET_TOOL_PARAMETER [TOOL=<name>] [T=<number>]  PARAMETER=parameter_<name> VALUE=<value>`: 
Change tool parameter in runtime.
Defaults to current tool if tool not specified.

### SAVE_TOOL_PARAMETER
`SAVE_TOOL_PARAMETER [TOOL=<name>] [T=<number>]  PARAMETER=parameter_<name>`: 
Saves the tool parameter to pending config changes.
Defaults to current tool if tool not specified.

### RESET_TOOL_PARAMETER
`RESET_TOOL_PARAMETER [TOOL=<name>] [T=<number>]  PARAMETER=parameter_<name> VALUE=<value>`: 
Resets a parameter to its original value.
Defaults to current tool if tool not specified.

### SET_TOOL_OFFSET
`SET_TOOL_OFFSET [TOOL=<name>] [T=<number>] [X=<float>] [Y=<float>] [Z=<float>]`
similar to 'SET_TOOL_PARAMETER' but for the gcode offsets.

### SAVE_TOOL_OFFSET
`SAVE_TOOL_OFFSET [TOOL=<name>] [T=<number>] [X=<float>] [Y=<float>] [Z=<float>]`
again, similar to 'SAVE_TOOL_PARAMETER' but for the gcode offsets.  
If no XYZ provided, saves currently active tool offsets to pending config changes.

# Gcodes if *fan* is specified for any of tools

### M106 
`M106 S<speed> [P<tool number>] [T<tool number>] [TOOL=<tool name>] `: Set fan speed. 
If P not specified sets speed for the current tool fan.
If `toolchanger.transfer_fan_speed` is enabled, current tool fan speed is transferred to the new tool on tool change.

### M107 
`M107 [P<tool number>] [T<tool number>] [TOOL=<tool name>] `: Stop fan.
With no parameters stops the current tool fan.

---

## Status

## tool

The following information is available in the `tool` object:
 - `name`: The tool name, eg 'tool T0'.
 - `tool_number`: The assigned tool number or -1 if not assigned.
 - `toolchanger`: The name of the toolchanger this tool is attached to. 
 - `extruder`: Name of the extruder used for this tool.
 - `fan`: Full name of the fan to be used as part cooling fan for this tool.
 - `active`: If this tool is currently the selected tool. 
 - `mounted_child`: The child tool which is currently mounted, or empty.
 - `params_*`: Set of values specified using params_*.
 - `gcode_x_offset`: current X offset.
 - `gcode_y_offset`: current Y offset.
 - `gcode_z_offset`: current Z offset.

## toolchanger

The following information is available in the `toolchanger` object:
 - `status`: One of 'uninitialized', 'ready', 'changing', 'error'.
 - `tool`: Name of currently selected/changed tool, or empty.
 - `tool_number`: Number of the currently selected tool, or -1.
 - `detected_tool`: Name of currently detected tool, or empty.
 - `detected_tool_number`: Number of the currently detected tool, or -1.
 - `tool_numbers`: List of assigned tool numbers, eg [0,1,2].
 - `tool_names`: List of tool names corresponding the assigned numbers.

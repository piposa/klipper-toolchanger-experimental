# Dock Location

The examples listed here are really only broad examples for now. one thing thats not shown/included right now is the 
`on_tool_mounted_gcode` and `on_tool_removed_gcode`.  
Those are especially helpful to keep the toolchangers internal state aligned with whatever tool is mounted.  
> *these do not run during the `select_tool` operation, neither do they during probing!*

```nunjucks
[toolchanger]
on_tool_mounted_gcode:
    {% if toolchanger.status|lower != 'error' %}
        INITIALIZE_TOOLCHANGER
        # equally keep the tool probe up to date if you want
        {% if printer.tool_probe_endstop is defined %}
            SET_ACTIVE_TOOL_PROBE T={detected_tool.tool_number}
        {% endif %}
    {% endif %}

on_tool_removed_gcode:
    {% if toolchanger.status|lower != 'error' %}
        INITIALIZE_TOOLCHANGER
    {% endif %}
```

Only manual/accidental mounts/removals will call this. ie, if a tool is dropped during printing you can prevent failures with:
```nunjucks
on_tool_removed_gcode:
    {% if printer.print_stats.state|lower == 'printing' %}
        PAUSE_BASE # or whatever your pause is called like, (its usually PAUSE_BASE when using mainsail)
    {% endif %}
```

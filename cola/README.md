# COLA Simulation Helper Files

## Transfer Function Generation
A Python script that generates the necessary transfer functions for running COLA simulations for w0wa cosmologies.

### Usage
1. Activate the `class_w0wa` environment
```
    conda activate class_w0wa
```

2. Run the script with
```
    python3 transfer_function_generator.py --input lhs.txt --path_to_save some/path/
```

Information is available using

```
    python3 transfer_function_generator.py --help
```

If no `--input` is provided, the script generates the transfer functions for the reference cosmology.

The save structure is in the following format:
```
    some/path/
    |---- 0/
    |     |---- transferinfo.dat
    |     |---- data_transfer_z0.000.dat
    |     |---- data_transfer_z0.020.dat
    |---- 1/
    |     |---- transferinfo.dat
    |     |---- data_transfer_z0.000.dat
    |     |---- data_transfer_z0.020.dat
    |---- 2/
    ...
```

## Lua File Generator
A Python script that generates Lua files for COLA settings.

### Usage
1. Run the script with
```
    python3 lua_file_generator.py --input lhs.txt --path_to_save some/path/
```

More information is available using

```
    python3 lua_file_generator.py --help
```
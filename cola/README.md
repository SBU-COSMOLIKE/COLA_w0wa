# Transfer Function Generator

## Usage
1. Activate the `class_w0wa` environment
```
    conda activate class_w0wa
```

2. Run the script with
```
    python3 transfer_function_generator.py --input lhs.txt --path_to_save some/path/
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
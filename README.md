# microplate: A class for manipulating microtiter plate data

## Use Cases

The goal of microplate is to make working with microplate data in Python
more convenient. Well data storage/retrieval is done through standard
microtiter plate labels ('A1' or 'P5') rather than through array indices.

## Features

  - Import directly from files without any needed manipulation
  - Arbitrary plate sizes up to 3456-wells.
  - Simplified retrieval of microplate data by using traditional well labels,
    rows/columns, or ranges.
  - Multiple data block support for assays with multiple reads.
  - Store arbitrary regions in plates to simplify calculating plate statistics.
  - Simple data normalization functions.
  - Iterator support to retrieve all data.
  - Easy view of plate data by printing the plate object.

## Source/Installation
The source code is currently hosted on GitHub at:
https://github.com/shumatejr/microplate

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/microplate/).

```sh
# PyPi
pip install microplate
```
Then just import the MTP class in your Python script.
```sh
from microplate import MTP
```

## Dependencies
- [NumPy](https://www.numpy.org)

## Example Usage
### Plate Creation
```sh
# Create an empty plate
plate_ratio = MTP(name = "Test Plate", rows=2, columns=3, blocks=3)

# Or define it from a file
# Multiple input files can be passed to add multiple data blocks to the plate.
# input_files input is in the format (filename, delimiter, file_row, file_column)
plate = MTP(
    name = "384-Well Test Plate", 
    rows = 16, 
    columns = 24,
    input_files = [ 
        ('test.txt',',',1,1),  # Comma delimited, Row 1, Column 1
        ('test.txt','\t',22,2), # Tab delimited, Row 22, Column 2
    ]
)

# View entire plate contents
print(plate)
```

### Well Manipulation/Retrieval
```sh
# Plate Access (empty string)
plate['']

# Well Access (well B3)
plate['B3'] = -1

# Row Access (row B)
plate['B']

# Column Access (column 3)
plate['3'] = 2

# Range Access
plate["A2:A3"] = -5

# If plates have multiple data blocks, add comma and value to specify block
plate['']    # Whole plate, implicitly data block 1
plate['',2]  # Whole plate, data block 2

plate["A2:A3",2] # Range from data block 2

# Values retrieved are numpy arrays, so 'matrix' math can be performed
# Make data block 3 a ratio of the other two data blocks
plate_ratio['',2] = 1
plate_ratio['',1] = 2

# Block 3 would store 1/2 in all wells
plate_ratio['',3] = plate_ratio['',2] / plate_ratio['',1]
```

### Regions
```sh
# Define regions
plate.set_region("high_ctrl", "A1:P1")
plate.set_region("low_ctrl", "A12:P12")
plate.set_region("full_plate", "A1:P12")

# Retrieve regions for statistics 
ctrl_window = plate.get_region("high_ctrl") / plate.get_region("low_ctrl")

# Plate Z' Calculation
high_avg = np.mean(plate.get_region("high_ctrl"))
low_avg = np.mean(plate.get_region("low_ctrl"))
high_std = np.mean(plate.get_region("high_ctrl"), ddof=1)
low_std = np.mean(plate.get_region("low_ctrl"), ddof=1)

z_prime = 1 - (3*high_std + 3*low_std)/abs(high_avg - low_avg)
```

### Normalization
```sh
# Normalize entire plate by zscore (returns a copy)
plate = plate.normalize_zscore()

# Normalize to a percent basis based on high/low control
plate_normalized = plate.normalize_percent(
    region_high = 'high_ctrl', 
    region_low = 'low_ctrl',
    method = 'median'
)

# All data blocks are normalized unless specified
# Scales data block 2 from 0-100%
plate_normalized = plate.normalize_percent(
    region_high = 'full_plate', 
    region_low = 'full_plate',
    block = 2,
    method = 'minmax'
)
```

### Iterator Support
```sh
# Sequential acess of all wells through an iterator
for ((row_index, column_index),label), well in plate(block=2):
    print(f"RowIndex:{row_index} ColumnIndex:{column_index} Label:{label} Data:{well}")
```

## License
[MIT](LICENSE)

## Credits
Developed by Justin Shumate















### 1.0
  - Initial Release.

### 1.1
  - Moved minmax into percent.

### 1.2
  - Refactored delete to be 'set 0'.
  - Added iterator support.
  - Added multiplier/invert to minmax normalization.
### 1.2.1
  - Added actual readme.

### 1.3
  - Significantly refactored all code, including adding new internal .
    functions to minimize replicated code.
  - Added row_to_index, index_to_row static methods.
  - Added add_block functionality to increase number of data blocks.

### 1.4
  - Added some standard plate statistic calculations (z, sw, drift).
  - Added formatter attribute to allow user editable formatting. For example, 
    pass "{: .2e}".format to get 2dec scientific notation to standardize length.
  - get_region can now be passed as wells or list of wells, bypassing set_region
    This is useful to turn a list of hits into a 1D list of data values.
  - Added a metadata dictionary (initialized with keys for each well) to store 
    properties user definable properties of each well. 
    For example: volumes, concentrations, ID's, hit_flags
  - Added calculate_cutoff_SD/excluded for cutoff calcs SD and SD w/o outliers.
  - Added get_hits to return a list of wells above (or below) input cutoff.

### TODO
  - Refactor normalize code















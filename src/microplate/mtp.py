"""Class definition for the microplate MTP class

A Microtiter plate class for python, simplifying an internal array to the way 
people usually reference/work with microtiter plates. Currently holds a name,
data array for numeric values, and metadata dictionary for properties per well. 
Supports arbitrary densities and stores arbitrary and non-contiguous regions.
"""
import re
import copy
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
from numpy.typing import NDArray

class MTP:
    """ A simplified way of working with microplate data in Python
    
    Parameters
    ----------
    rows : int
        Number of rows to store in the resulting microplate.
    columns : int
        Number of columns to store in the resulting microplate.
    blocks : int, optional
        Number of datablocks in the resulting microplate. Defaults to 1.
    input_files : List of Tuple, optional
        Optional input to create a microplate from a file instead of manually
        defining the plate.  If multiple Tuples are included in the list, there
        is a block created for each Tuple.
        
        Tuple is of the form (file_path, delimiter, file_row, file_column)
        file_path : str
            Path from the working directory of the file the data is located in.
        delimiter : str
            File delimiter inside the file pointed to by file_path.
        file_row : int
            Starting row (counting from 1) where the data block is located.
        file_column : int
            Starting column (counting from 1) where the data block is located.
    plate_metadata : List of Tuple, optional
        Similar to the input_files argument, except this scans for individual
        pieces of data in a file to store in the metadata dictionary.
        
        Tuple is of the form (file_path, delimiter, file_row, file_column, key)
        key : str
            This is the top-level key that will be created in the metadata dict.
        Remaining elements of tuple are the same as in the input_files tuple.
    metadata_keys : Dict of Str, Any, optional
            Default keys for well-level data defined in the metadata dictionary.
       
    Attributes
    ----------
    name : str
        Storage for string representation of microplate (eg barcode).
    regions : Dict[str|List[str]]
        Contains region names which are arbitrary sections of the plate and
        the wells that are in each region. Regions can be set as a list of
        individual wells ['A1','A5','B3'], or as ranges ['A2:B6']
    formatter : str
        Formatter for the __str__ representation which can be overwridden. For
        example, set equal to "{: .2f}".format to print as floats with 2 dec.
    metadata : Dict
        Dictionary initialized with a key for each well. Used for storing any
        relevant metadata for the plate, well-level or plate-level.
    __rows : int
        Number of rows to store in the resulting microplate.
    __columns : int
        Number of columns to store in the resulting microplate.
    blocks : int, optional
        Number of datablocks in the resulting microplate. Defaults to 1.
    __data : 3-dimensional Numpy NDArray
        Internal representation of the microplate data. Stored as a 3D Matrix
        where each slice of the matrix is a separate data block.
    __index : Tuple[int, int]
        Numpy internal representation of matrix index for iteration.
    __iter : iterator
        Numpy iterator for traversing matrix.
    __iterate: bool
        Stores whether iterator should continue to next value.
    """
    
    def __init__(self, rows: int, columns: int, 
                 blocks: int = 1, name: str = "plate", 
                 input_files: List[Tuple[str, str, int, int]] = None,
                 plate_metadata: List[Tuple[str, str, int, int, str]] = None,
                 metadata_keys: Dict[str, Any] = {}):
        
        if rows <= 0 or columns <= 0:
            raise ValueError("Plate dimensions must be greater than zero.")
        
        self.__rows = rows
        self.__cols = columns
        self.regions = {}
        
        self.name = name
        self.formatter = "{: .2f}".format
        
        # Initialize metadata dictionary and parse plate-level metadata
        self.metadata = {}
        if plate_metadata is not None:
            for meta_tuple in plate_metadata:
                self.metadata[meta_tuple[-1]] = self._parse_metadata(meta_tuple)
        # Initialize key for each well for well-level metadata
        # Class accepts input of default keys for each well
        for row in range(1, self.__rows+1):
            for col in range(1, self.__cols+1):
                self.metadata[MTP.index_to_row(row) + str(col)] = metadata_keys
        
        # If no input is specified, create a microplate pre-filled with zeros
        if input_files is None:
            self.blocks = blocks
            self.__data = np.zeros((self.blocks, self.__rows, self.__cols))
        # In this instance, parse files to get contents of blocks
        else:
            # Support for multiple data blocks inside a file
            self.blocks = len(input_files)
            self.__data = np.zeros((self.blocks, self.__rows, self.__cols))
            
            for block_num, data_tuple in enumerate(input_files):
                self.__data[block_num] = self._parse_data(data_tuple)
        
        # Iterator defaults to none because it breaks deepcopy otherwise
        self.__index = None
        self.__iter = None
        self.__iterate = True
        self.__iterblock = 0
    
    # Internal method for parsing data from files during object construction
    def _parse_file(self, file_path: str, delimiter: str, file_row: int, 
                    file_column: int, parse_type: str = None) -> NDArray:
        
        # Differentiate between parsing data or metadata, with readable errors
        if parse_type is None:
            num_rows = self.__rows
            num_cols = self.__cols
            parse_type = "Input Data"
        else: num_rows = num_cols = 1
        
        with Path(file_path).open("r") as file:
            lines = file.readlines() # Not great for very large files
            
            # Loop instead of list comprehension because it becomes unreadable
            if len(lines) < file_row-1 + num_rows:
                raise ValueError(f"{parse_type} row is invalid.")
            lines = lines[file_row-1 : file_row-1 + num_rows]
            
            for idx, line in enumerate(lines):
                line = line.strip().split(delimiter) # Remove newlines
                if len(line) < file_column-1 + num_cols:
                    raise ValueError(f"{parse_type} column is invalid.")
                lines[idx] = line[file_column-1 : file_column-1 + num_cols]
        return lines
    def _parse_data(self, data_tuple):
        return np.array(self._parse_file(*data_tuple))
    def _parse_metadata(self, metadata_tuple):
        return self._parse_file(*metadata_tuple)[0][0] # No list
    
    # Internal checks for get/set
    def _key_check(self, key) -> Tuple[str, int]:
        # By default operate on the first block
        block_num = 0
        
        # If tuple passed, first part is key and second is block_num
        if type(key) is tuple:
            if (len(key) == 2 and isinstance(key[1], int) 
                and key[1] > 0 and key[1] <= self.blocks):
                # Well is first element of tuple, block_num second
                key, block_num = key
                block_num -= 1
            elif len(key) == 1:
                key = key[0]
                block_num = slice(0, self.blocks)
            else:
                raise ValueError(f"Invalid Input {key}")
        # Ensure key is a str
        if type(key) is not str:
            raise TypeError(f"{key} must be of type str")
        # Ensure no invalid characters in key
        if re.search("[^A-Z0-9:]", key):
            raise ValueError(f"Invalid characters in {key}")
        
        return key, block_num
    
    # Convert an input well representation into indices
    def _well_transform(self, key: str) -> Tuple[int, int, int, int]:
        
        row_arr = [0, self.__rows, None]
        col_arr = [0, self.__cols, None]
        
        for index, value in enumerate(key.split(":")):
            row = re.search("[A-Z]+", value)
            col = re.search("[0-9]+", value)
            
            if row: row_arr[index] = MTP.row_to_index(row.group())-1 + index
            if col: col_arr[index] = int(col.group())-1 + index
        
        # For single well searches, end of slice needs to be incremented
        if row: row_arr[index+1] = row_arr[index] + 1
        if col: col_arr[index+1] = col_arr[index] + 1
        
        return row_arr[0], row_arr[1], col_arr[0], col_arr[1]
    # Convert indices to slices
    def _well_transform_slice(self, key: str) -> Tuple[object, object]:
        row1, row2, col1, col2 = self._well_transform(key)
        return slice(row1, row2), slice(col1, col2)
    # Convert indices to list of wells
    def _well_transform_list(self, key: str) -> List[str]:
        row1, row2, col1, col2 = self._well_transform(key)
        return [MTP.index_to_row(row) + str(col) 
                for row in range(row1+1, row2+1) 
                for col in range(col1+1, col2+1)]
    
    # Overloaded list operations for retrieving/setting microplate data
    def __getitem__(self, key) -> NDArray:
        wells, block_num = self._key_check(key)
        rows, cols = self._well_transform_slice(wells)
        return self.__data[block_num, rows, cols]
    def __setitem__(self, key, value):
        wells, block_num = self._key_check(key)
        rows, cols = self._well_transform_slice(wells)
        self.__data[block_num, rows, cols] = value
    def __delitem__(self, key):
        self.__setitem__(key, 0)
    
    # Custom string representation of a MTP for printing
    def __str__(self):
        row_matrix = np.array2string(
            self.__data + 0.0, # 0.0 is added to eliminate -0.0 from display
            formatter={"float_kind": self.formatter}, 
            max_line_width = np.inf, 
            threshold=np.inf,
        )
        row_matrix = re.sub(r"\[|\]", " ", row_matrix) # Remove brackets
        
        return (f"{self.name}\n#Blocks:{self.blocks} #Rows:{self.__rows} " 
                f"#Columns:{self.__cols}\n{row_matrix}\n")
    
    # Overloaded operations for iterator support of plate (uses numpy.nditer)
    def __call__(self, block: int = 0):
        if block < 0:
            ValueError("Invalid Block number")
        self.__iter = np.nditer(self.__data[block-1], flags=["multi_index"])
        self.__iterblock = block
        return self
    def __iter__(self):
        return self
    def __next__(self):
        if not self.__iterate:
            # Reset iterator to default values to enable deepcopy afterwards
            self.__index = None
            self.__iter = None
            self.__iterate = True
            self.__iterblock = 0
            raise StopIteration
        
        # Pull row and column from the iterator and calculate the label
        self.__index = self.__iter.multi_index # Numpy internal tuple
        well_row = self.__index[0]+1
        well_col = self.__index[1]+1
        well = f"{MTP.index_to_row(well_row)}{well_col}"
        
        # well_value can be a single element or a list
        if self.__iterblock != 0:
            well_value = self.__iter[0].item()
        else:
            well_value = []
            for block_num in range(self.blocks):
                well_value.append(
                    float(self.__data[block_num][well_row-1][well_col-1])
                )
        
        # iternext() automatically advances iterator, so call it last and store
        self.__iterate = self.__iter.iternext()
        
        # Output of the form "A1", 1, 1, Value
        return well, well_row, well_col, well_value
    
    def set_region(self, name: str, wells: str|List[str]):
        """Add a defined region to the microplate.
        
        Regions can be individual wells 'A1', ranges 'A1:B2', or lists
        of either/or such as ['A1','A2'] or ['A1:P1','A24:P24'] or even
        ['A1', 'A24:P24']. 
        
        Regions are valid for any block of the microplate as the block is only
        specified on the get method.
        
        Type for wells is enforced and individual items are converted to a list.
        
        Parameters
        ----------
        name : str
            Specified name of the region which will be used for retrieval.
        wells : str|List[str]
            Wells contained within the region.
        
        Raises
        ------
        TypeError
            If wells is not str or a List[str].
        ValueError
            Improper characters found in wells.
        """
        
        # Enforce type of str or List[str], individual str is converted to list
        if type(wells) is str: wells = [wells]
        if type(wells) is list and all(type(well) is str for well in wells):
            
            # Check if all the sections of the region have valid characters
            if any(re.search("[^A-Z0-9:]", well) for well in wells):
                raise ValueError(f"Invalid well label in {wells}.")
            else:
                self.regions[name] = wells
        else:
            raise TypeError(f"{wells} must be of type str or list[str].")
    
    def get_region(self, name: str = None, block: int = 1,
                   wells: str|list[str] = None) -> NDArray:
        """Retrieve a region as a 1D array.
        
        Parameters
        ----------
        name : str, optional
            Specified name of the region which will be used for retrieval. name
            is optional if wells are passed instead.
        wells : str, optional
            List of wells to get as 1D, instead of using a specified name. This
            allows for a region to be retrieved without being set by name.
        block : int, optional
            Data block number to use when retrieving wells.
        
        Returns
        -------
        NDArray
            1D Numpy array of wells specified in the region.
        
        Raises
        ------
        ValueError
            If both name and wells are not set.
        """
        # Check if block accessed is within the range
        if block < 1 or block > self.blocks:
            raise ValueError("Invalid block number.")
        
        # Retrieve the wells in the region (this will fail for invalid region)
        if type(wells) is str: wells = [wells]
        if type(wells) is list and all(type(well) is str for well in wells):
            well_list = wells
        elif name: well_list = self.regions[name]
        else: raise ValueError("Invalid region input")
        
        value_list = []
        # Parse each region section and add it to the output list
        for wells in well_list:
            rows, cols = self._well_transform_slice(wells)
            value_list.extend(self.__data[block-1,rows,cols].flatten().tolist())
        
        return np.array(value_list)
    
    def get_labels(self, region: str|List[str]) -> List[str]:
        """Retrieve a region as a 1D array.
        
        Parameters
        ----------
        region : str|List[str]
            Specified region to retrieve the wells from, or a list of wells
            which could be used to create a region.
        
        Returns
        -------
        List[str]
            List of individual wells representing the region.
        
        Raises
        ------
        ValueError
            If the region name is not valid.
        TypeError
            If the input is not str or List[str].
        """
        if type(region) is str:
            if region not in self.regions:
                raise ValueError("Invalid region input")
            region_wells = self.regions[region]
        elif type(region) is list and all(type(well) is str for well in region):
            region_wells = region
        else: raise ValueError("Wells are not List[str]")
        
        well_list = []
        for well in region_wells:
            well_list.extend(self._well_transform_list(well))
        return well_list
    
    # Given a cutoff and a region, return the wells which are 
    def get_hits(self, region: str, cutoff: float|int, 
                 block: int = 1, negative_cutoff: bool = False) -> List[str]:
        """Return a list of wells that meet the input cutoff criteria
        
        Parameters
        ----------
        region : str
            Region from which the wells will be selected.
        cutoff : float
            Cutoff used for hit selection, finding wells > cutoff
        block : int, optional
            Data block where data values will be compared to cutoff
        negative_cutoff : bool, optional
            Use a negative cutoff instead, finding values < cutoff
            
        Raises
        ------
        TypeError
            Cutoff provided is not of type float or int.
        """
        if type(cutoff) is not float and type(cutoff) is not int:
            raise TypeError("Cutoff must be of type int or float")
        
        # Create a list of wells in input region
        well_list = self.get_labels(region)
        
        # Create a list of hits across entire plate
        if not negative_cutoff:
            row_indices, col_indices = np.nonzero(self.__data[block-1] > cutoff)
        else:
            row_indices, col_indices = np.nonzero(self.__data[block-1] < cutoff)
        hit_list = [MTP.index_to_row(int(row+1)) + str(col+1) 
                    for row, col in tuple(zip(row_indices,col_indices))]
        
        # Return intersection of list of hits and list of wells in region
        return list(set(well_list) & set(hit_list))
    
    def add_block(self, num_blocks: int = 1):
        """Add a new empty block to the microplate.
        
        Parameters
        ----------
        num_blocks : int, optional
            Add num_blocks data blocks to the plate.
            
        Raises
        ------
        ValueError
            Num_blocks is not an int or is <= 0
        """
        if type(num_blocks) is not int or num_blocks <= 0:
            raise ValueError(f"Improper number of blocks {num_blocks}")
        
        self.blocks += num_blocks
        self.__data.resize(self.blocks, self.__rows, self.__cols)
    
    
    # IMPROVEMENT: repeated code in normalizations, create internal method?
    def normalize_zscore(self, region_name: str = None, 
                         block: int = 0, df: int = 1):
        """Create a new MTP that is normalized by z-score
        
        Parameters
        ----------
        region_name : str, optional
            Name of region used for normalization. If none specified, this will
            use the entire plate as the region to normalize to.
        block : int, optional
            Data block number to use when retrieving wells. If none specified,
            then all blocks of the MTP are normalized on a bock-by-block basis.
        df : int, optional
            Numpy degrees of freedom for standard deviation calculation.
        
        Returns
        ------
        MTP
            A copy of the MTP with the normalization applied.
        
        Raises
        ------
        ValueError
            If region does not exist, or if block is incorrect.
        """
        # Verify user input
        if type(block) is not int or block > self.blocks:
            raise ValueError("Invalid block number")
        
        norm_plate = copy.deepcopy(self)
        
        # If no block is specified, perform normalization on by-block basis
        if block <= 0:
            block_range = range(self.blocks)
        # Otherwise, only normalize specified block
        else:
            block_range = [block-1]
        
        for block_num in block_range:
            # Normalize to the entire plate
            if region_name is None:
                norm_plate.__data[block_num] = (
                    (
                        norm_plate.__data[block_num] 
                        - np.mean(norm_plate.__data[block_num])
                    )
                    / np.std(norm_plate.__data[block_num], ddof=1)
                )
            else:
                norm_plate.__data[block_num] = (
                    (
                        norm_plate.__data[block_num] 
                        - np.mean(self.get_region(region_name, block_num+1))
                    )
                    / np.std(self.get_region(region_name, block_num+1), ddof=df)
                )
        return norm_plate
    
    def normalize_percent(self, region_high: str, region_low: str, 
                          block: int = 0, method: str = "median", 
                          multiplier: int = 100, invert: bool = False):
        """Create a new MTP that is normalized on a percentage scale
        
        Parameters
        ----------
        region_high : str
            Name of region used for 100% in normalization.
        region_low : str
            Name of region used for 0% in normalization.
        block : int, optional
            Data block number to use when retrieving wells. If none specified,
            then all blocks of the MTP are normalized on a bock-by-block basis
        method : str, optional
            Statistic used for normalization. Options include: mean, median, and
            minmax.
        multiplier : int, optional
            Scaling factor used after normalization. Defaults to 100 for percent
            normalization, but can be changed to 1 for 0->1 normalization.
        invert : bool, optional
            Subtract arr val from multiplier. Only usable in minmax method.
        
        Returns
        ------
        MTP
            A copy of the MTP with the normalization applied.
        
        Raises
        ------
        ValueError
            If regions do not exist, method does not exist, or block incorrect.
        """
        # Verify user input
        if type(block) is not int or block > self.blocks:
            raise ValueError
        
        # If no block is specified, perform normalization on by-block basis
        if  block <= 0:
            block_range = range(self.blocks)
        # Otherwise, only normalize specified block
        else:
            block_range = [block-1]
        
        norm_plate = copy.deepcopy(self)
        
        for block_num in block_range:
            if method == "median":
                norm_plate.__data[block_num] = multiplier * (
                    (
                        norm_plate.__data[block_num] 
                        - np.median(self.get_region(region_low, block_num+1))
                    )
                    /(
                        np.median(self.get_region(region_high, block_num+1)) 
                        - np.median(self.get_region(region_low, block_num+1))
                    )
                )
            elif method == "mean":
                norm_plate.__data[block_num] = multiplier * (
                    (
                        norm_plate.__data[block_num] 
                        - np.mean(self.get_region(region_low, block_num+1))
                    )
                    /(
                        np.mean(self.get_region(region_high, block_num+1)) 
                        - np.mean(self.get_region(region_low, block_num+1))
                    )
                )
            elif method == "minmax":
                norm_plate.__data[block_num] = multiplier * (
                    (
                        norm_plate.__data[block_num] 
                        - np.min(self.get_region(region_low, block_num+1))
                    )
                    /(
                        np.max(self.get_region(region_high, block_num+1)) 
                        - np.min(self.get_region(region_low, block_num+1))
                    )
                )
                if invert:
                    norm_plate.__data[block_num] = (
                        multiplier - norm_plate.__data[block_num]
                    )
            else:
                raise ValueError("Invalid Method")
        
        return norm_plate
    
    # Routine plate statistic calculations
    def calc_z(self, region_high: str, region_low: str, 
               block: int = 1, df: int = 1) -> float:
        """Calculate the z statistic for input regions.
        
        Parameters
        ----------
        region_high : str
            Upper region for z-calculation.
        region_low : str
            Lower region for z calculation.
        block : int, optional
            Block number to perform the calculation on (default 1).
        df : int, optional
            Degrees of freedom for standard deviation calculation (default 1).
        Returns
        -------
        float
            Calculated window based on input regions and method.
        
        Raises
        ------
        ValueError
            Invalid method of input chosen (valid options 'median' or 'mean')
        """
        region_high = self.get_region(region_high, block)
        region_low = self.get_region(region_low, block)
        
        avg_high = np.mean(region_high)
        avg_low = np.mean(region_low)
        std_high = np.std(region_high, ddof=df)
        std_low = np.std(region_low, ddof=df)
        
        return float(1 - ((3*std_high + 3*std_low) / abs(avg_high - avg_low)))
    
    def calc_sw(self, region_high: str, region_low: str, 
                block: int = 1, method: str = "median") -> float:
        """Calculate the signal window between two regions.
        
        Parameters
        ----------
        region_high : str
            Numerator region for window.
        region_low : str
            Denominator region for window.
        block : int, optional
            Block number to perform the calculation on (defaults to 1).
        method : str, optional
            Statistic to use for calculating the window ('median' or 'mean').
        Returns
        -------
        float
            Calculated window based on input regions and method.
        
        Raises
        ------
        ValueError
            Invalid method of input chosen (valid options 'median' or 'mean')
        """
        region_high = self.get_region(region_high, block)
        region_low = self.get_region(region_low, block)
        
        if method == "median":
            high = np.median(region_high)
            low = np.median(region_low)
        elif method == "mean":
            high = np.mean(region_high)
            low = np.mean(region_low)
        else:
            raise ValueError(f"Invalid input {method} for method.")
        return float(high/low) if high/low >= 1 else float(low/high)
    
    def calc_drift(self, region: str, block: int = 1) -> Tuple[float, float]:
        """Calculate the drift across rows/columns for a given region.
        
        Parameters
        ----------
        region : str
            Region to calculate drift against. Only considers first defined
            section of region in non-contiguous regions.
        block : int, optional
            Block number to perform the calculation on (defaults to 1).
        
        Returns
        -------
        Tuple[float,float]
            Calculated window based on input regions and method.
        """
        data_matrix = self.__getitem__((self.regions[region][0], block))
        num_rows, num_cols = data_matrix.shape
        
        row_max = -np.inf
        row_min = np.inf
        for row_index in range(num_rows):
            row_median = np.median(data_matrix[row_index,:])
            if row_median > row_max: row_max = row_median
            if row_median < row_min: row_min = row_median
            
        col_max = -np.inf
        col_min = np.inf
        for col_index in range(num_cols):
            col_median = np.median(data_matrix[:,col_index])
            if col_median > col_max: col_max = col_median
            if col_median < col_min: col_min = col_median
        
        return float(row_max-row_min), float(col_max-col_min)
    
    # Internal cutoff calculation used by cutoff methods
    def _cutoff(self, region_data: List[float], 
                df: int = 1, num_deviations: int = 3,
                negative_cutoff: bool = False) -> float:
        
        data_avg = np.mean(region_data)
        data_sd = np.std(region_data, ddof=df)
        if negative_cutoff: data_sd *= -1
        
        return float(data_avg + num_deviations*data_sd)
    
    def calc_cutoff_sd(self, region: str, block: int = 1, df: int = 1, 
                       num_deviations: int = 3, 
                       negative_cutoff: bool = False) -> float:
        """Calculate cutoff using a regions average and stdev.
        
        Parameters
        ----------
        region : str
            Region from which the wells will be selected.
        block : int, optional
            Data block where data values will be compared to cutoff
        df : int, optional,
            Degrees of freedom for standard deviation calculation (default 1).
        devs : int, optional
            Number of standard deviations used for cutoff (default 3).
        negative_cutoff : bool, optional
            Return avg-3sd instead of avg+3sd
            
        Returns
        -------
        float
            AVG+3SD or AVG-3SD, depending on negative_cutoff passed
        """
        region_data = self.get_region(region, block)
        return self._cutoff(region_data, df, num_deviations, negative_cutoff)
    
    def calc_cutoff_excluded(self, region: str, 
                             region_high: str, region_low: str, block: int = 1, 
                             df: int = 1, num_deviations: int = 3, 
                             negative_cutoff: bool = False) -> float:
        """Calculate cutoff using a regions average and stdev (remove outliers).
        
        Wells that are > avg+3sd region_high or < avg-3sd region_low are not
        included in the cutoff calculation.
        
        Parameters
        ----------
        region : str
            Region from which the wells will be selected.
        region_high : str
            Exclude wells above num_devs region_high from cutoff calculation.
        region_high : str
            Exclude wells below num_devs region_low from cutoff calculation.
        block : int, optional
            Data block where data values will be compared to cutoff
        df : int, optional,
            Degrees of freedom for standard deviation calculation (default 1).
        devs : int, optional
            Number of standard deviations used for cutoff (default 3).
        negative_cutoff : bool, optional
            Return avg-3sd instead of avg+3sd
            
        Returns
        -------
        float
            AVG+3SD or AVG-3SD, depending on negative_cutoff passed
        """
        cutoff_high = self.calc_cutoff_sd(region=region_high, block=block, 
                                          df=df, num_deviations=num_deviations)
        cutoff_low = self.calc_cutoff_sd(region=region_low, block=block, 
                                         df=df, num_deviations=num_deviations,
                                         negative_cutoff=True)
        
        region_data = [value for value in self.get_region(region, block)
                         if value <= cutoff_high and value >= cutoff_low]
        if len(region_data) == 0:
            raise ValueError("No wells meet excluded cutoff critiera")
        
        return self._cutoff(region_data, df, num_deviations, negative_cutoff)
        
    # Convenient methods for converting row labels to index and vice versa
    @staticmethod
    def row_to_index(row: str) -> int:
        """Convert a row label to a row number (starting with 1).
        A -> 1; AA -> 27; AF -> 32
        
        Parameters
        ----------
        row : str
            Row label
        
        Returns
        -------
        int
            Integer representation of passed row string.
        
        Raises
        ------
        ValueError
            Input is not a letter.
        """
        row = str(row) # Convert to str representation
        if re.search("[^A-Z]", row, re.IGNORECASE):
            raise ValueError(f"Invalid characters in {row}")
        
        # Treat row label as base-26 and convert to base-10 int
        sum = 0
        for pos, ch in enumerate(reversed(row.upper())):
            sum += (ord(ch)-ord('A')+1) * (26 ** pos)
        return sum
    @staticmethod
    def index_to_row(row_index: int) -> str:
        """Convert a row number (starting with 1) to a row label.
        1 -> A; 27 -> AA; 32 -> AF
        
        Parameters
        ----------
        row_index : int
            Row label
        
        Returns
        -------
        str
            String representation of input row_index.
        
        Raises
        ------
        TypeError
            Input is not an integer.
        """
        if type(row_index) is not int:
            raise TypeError(f"{row_index} is not of type int.")
        
        row = ""
        while row_index > 0:
            remainder = (row_index-1) % 26
            row_index = (row_index-1) // 26
            row = chr(remainder + ord('A')) + row
        return row
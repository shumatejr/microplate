"""Class definition for the microplate MTP class

A Microtiter plate class for python, simplifying an internal array to the way 
people usually reference/work with microtiter plates. Currently holds a name, 
supports arbitrary densities and stores arbitrary and non-contiguous regions.
"""
import re
import sys
import copy
from typing import Tuple, List

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
            File delimiter inside the file pointed to by file_path
        file_row : int
            Starting row (counting from 1) where the data block is located.
        file_column : int
            Starting column (counting from 1) where the data block is located.

    Attributes
    ----------
    name: str
        Storage for string representation of microplate (eg barcode).
    __rows : int
        Number of rows to store in the resulting microplate.
    __columns : int
        Number of columns to store in the resulting microplate.
    __blocks : int, optional
        Number of datablocks in the resulting microplate. Defaults to 1.
    __regions : Dict[str|List[str]]
        Contains region names which are arbitrary sections of the plate and
        the wells that are in each region. Regions can be set as a list of
        individual wells ['A1','A5','B3'], or as ranges ['A2:B6']
    __arr : 3-dimensional Numpy NDArray
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
                 input_files: List[Tuple[str, str, int, int]] = None,):
        
        self.__rows = rows
        self.__cols = columns
        
        self.name = name
        self.__regions = {}
        
        # If no input is specified, create a microplate pre-filled with zeros
        if input_files is None:
            self.__blocks = blocks
            self.__arr = np.zeros((self.__blocks, self.__rows, self.__cols))
        # In this instance, parse files to get contents of blocks
        else:
            # Support for multiple data blocks inside a file
            self.__blocks = len(input_files)
            self.__arr = np.zeros((self.__blocks, self.__rows, self.__cols))
            
            for block_num, data_block in enumerate(input_files):
                self.__arr[block_num] = np.array(self._parse_file(*data_block))
        
        # Iterator defaults to none because it breaks deepcopy otherwise
        self.__index = None
        self.__iter = None
        self.__iterate = True
    
    # Internal method for parsing data from files during object construction
    def _parse_file(self, file_path: str, delimiter: str, 
                     file_row: int, file_column: int) -> NDArray:
        
        # Initialize np matrix of appropriate size to store our data
        data = np.zeros((self.__rows,self.__cols))
        
        # Open the file at the file path
        with open(file_path, 'r') as file:
            # Iterate through the files rows to the desired line number
            for line_number, line in enumerate(file, start=1):
                # Only consider rows within range of file_row to size of data
                if (line_number >= file_row and 
                    line_number <  file_row + self.__rows):
                    
                    # Break up the line by the specified delimiter
                    split_line = line.split(delimiter)
                    
                    # Iterate through columsn to the desired column number
                    for column_number, value in enumerate(split_line, start=1):
                        if (column_number >= file_column and 
                            column_number <  file_column + self.__cols):
                            
                            # Fail if non-numeric input is found in file
                            try:
                                data[line_number-file_row, 
                                     column_number-file_column] = value
                            except ValueError:
                                print(f"Invalid input in file {file_path}." 
                                      f"Check delimiter/block location.")
                                sys.exit(1)
        return data
    
    # Internal checks for get/set
    def _key_check(self, key) -> Tuple[str, int]:
        # By default operate on the first block
        block_num = 0
        
        # If tuple passed, first part is key and second is block_num
        if type(key) is tuple:
            if (len(key) == 2 and isinstance(key[1], int) 
                and key[1] > 0 and key[1] <= self.__blocks):
                # Well is first element of tuple, block_num second
                key, block_num = key
                block_num -= 1
            else:
                raise ValueError(f"Invalid Input {key}")
        # Ensure key is a str
        if not type(key) is str:
            raise TypeError(f"{key} must be of type str")
        # Ensure no invalid characters in key
        if re.search("[^A-Z0-9:]", key):
            raise ValueError(f"Invalid characters in {key}")
        
        return key, block_num
    
    # Convert an input well representation into equivalent array slices
    def _well_transform(self,key: str) -> Tuple[object, object]:
        
        row_arr = [0, self.__rows, None]
        col_arr = [0, self.__cols, None]
        
        for index, value in enumerate(key.split(":")):
            row = re.search("[A-Z]{1,2}", value)
            col = re.search("[0-9]{1,2}", value)
            
            if row: row_arr[index] = self.row_to_index(row.group())-1 + index
            if col: col_arr[index] = int(col.group())-1 + index
        
        # For single well searches, end of slice needs to be incremented
        if row: row_arr[index+1] = row_arr[index] + 1
        if col: col_arr[index+1] = col_arr[index] + 1
        
        return slice(row_arr[0], row_arr[1]), slice(col_arr[0], col_arr[1])
    
    # Overloaded list operations for retrieving/setting microplate data
    def __getitem__(self, key) -> NDArray:
        wells, block_num = self._key_check(key)
        rows, cols = self._well_transform(wells)
        return self.__arr[block_num, rows, cols]
    def __setitem__(self, key, value):
        wells, block_num = self._key_check(key)
        rows, cols = self._well_transform(wells)
        self.__arr[block_num, rows, cols] = value
    def __delitem__(self, key):
        self.__setitem__(key, 0)
    
    # Custom string representation of a MTP for printing
    def __str__(self):
        # Two decimal limit on print
        row_matrix = np.array2string(
            self.__arr + 0.0, # 0.0 is added to eliminate -0.0 from display
            formatter={"float_kind": lambda x: "%.2f" % x}, 
            max_line_width = 99999, 
            threshold=1536,
        )
        return (f"{self.name}\n#Blocks:{self.__blocks} #Rows:{self.__rows} " 
                f"#Columns:{self.__cols}\n{row_matrix}\n")
    
    # Overloaed operations for iterator support of plate (uses numpy.nditer)
    def __call__(self, block: int = 1):
        self.__iter = np.nditer(self.__arr[block-1], flags=["multi_index"])
        return self
    def __iter__(self):
        return self
    def __next__(self):
        if not self.__iterate:
            # Reset iterator to default values to enable deepcopy afterwards
            self.__index = None
            self.__iter = None
            self.__iterate = True
            raise StopIteration
        
        # iternext() automatically advances iterator, so call it last and store
        self.__index = self.__iter.multi_index # Numpy internal tuple
        well_value = self.__iter[0].item()
        self.__iterate = self.__iter.iternext()
        
        # Output of the form "A1", 1, 1, Value
        well_row = self.__index[0]+1
        well_col = self.__index[1]+1
        well = f"{self.index_to_row(well_row)}{well_col}"
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
        if type(wells) is str:
            wells = [wells]
        if type(wells) is list and all(type(well) is str for well in wells):
            
            # Check if all the sections of the region have valid characters
            if any(re.search("[^A-Z0-9:]", well) for well in wells):
                raise ValueError(f"Invalid well label in {wells}.")
            else:
                self.__regions[name] = wells
        else:
            raise TypeError(f"{wells} must be of type str or list[str].")
    
    def get_region(self, name: str, block: int = 1) -> NDArray:
        """Retrieve a previously set region as a 1D array.
        
        Parameters
        ----------
        name : str
            Specified name of the region which will be used for retrieval.
        block : int, optional
            Data block number to use when retrieving wells.
        
        Returns
        -------
        NDArray
            1D Numpy array of wells specified in the region.
        """
        # Check if block accessed is within the range
        if block < 1 or block > self.__blocks:
            raise ValueError("Invalid block number.")
        
        # Retrieve the wells in the region (this will fail for invalid region)
        well_list = self.__regions[name]
        
        value_list = []
        # Parse each region section and add it to the output list
        for wells in well_list:
            rows, cols = self._well_transform(wells)
            value_list.extend(self.__arr[block-1,rows,cols].flatten().tolist())
        
        return np.array(value_list)
    
    def add_block(self, num_blocks: int = 1):
        """Create a new MTP that is normalized by z-score
        
        Parameters
        ----------
        num_blocks : int, optional
            Add num_blocks data blocks to the plate.
            
        Raises
        ------
        ValueError
            Num_blocks is not an int or is <= 0
        """
        if not type(num_blocks) is int or num_blocks <= 0:
            raise ValueError(f"Improper number of blocks {num_blocks}")
        
        self.__blocks += num_blocks
        self.__arr.resize(self.__blocks, self.__rows, self.__cols)
    
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
        if not type(block) is int or block > self.__blocks:
            raise ValueError("Invalid block number")
        
        norm_plate = copy.deepcopy(self)
        
        # If no block is specified, perform normalization on by-block basis
        if  block <= 0:
            block_range = range(self.__blocks)
        # Otherwise, only normalize specified block
        else:
            block_range = [block-1]
        
        for block_num in block_range:
            # Normalize to the entire plate
            if region_name == None:
                norm_plate.__arr[block_num] = (
                    (
                        norm_plate.__arr[block_num] 
                        - np.mean(norm_plate.__arr[block_num])
                    )
                    / np.std(norm_plate.__arr[block_num], ddof=1)
                )
            else:
                norm_plate.__arr[block_num] = (
                    (
                        norm_plate.__arr[block_num] 
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
        if not type(block) is int or block > self.__blocks:
            raise ValueError
        
        # If no block is specified, perform normalization on by-block basis
        if  block <= 0:
            block_range = range(self.__blocks)
        # Otherwise, only normalize specified block
        else:
            block_range = [block-1]
        
        norm_plate = copy.deepcopy(self)
        
        for block_num in block_range:
            if method == "median":
                norm_plate.__arr[block_num] = multiplier * (
                    (
                        norm_plate.__arr[block_num] 
                        - np.median(self.get_region(region_low, block_num+1))
                    )
                    /(
                        np.median(self.get_region(region_high, block_num+1)) 
                        - np.median(self.get_region(region_low, block_num+1))
                    )
                )
            elif method == "mean":
                norm_plate.__arr[block_num] = multiplier * (
                    (
                        norm_plate.__arr[block_num] 
                        - np.mean(self.get_region(region_low, block_num+1))
                    )
                    /(
                        np.mean(self.get_region(region_high, block_num+1)) 
                        - np.mean(self.get_region(region_low, block_num+1))
                    )
                )
            elif method == "minmax":
                norm_plate.__arr[block_num] = multiplier * (
                    (
                        norm_plate.__arr[block_num] 
                        - np.min(self.get_region(region_low, block_num+1))
                    )
                    /(
                        np.max(self.get_region(region_high, block_num+1)) 
                        - np.min(self.get_region(region_low, block_num+1))
                    )
                )
                if invert:
                    norm_plate.__arr[block_num] = (
                        multiplier - norm_plate.__arr[block_num]
                    )
            else:
                raise ValueError("Invalid Method")
        
        return norm_plate
    
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
        if not type(row_index) is int:
            raise TypeError(f"{row_index} is not of type int.")
        
        row = ""
        while True:
            remainder = (row_index-1) % 26
            row_index = (row_index-1) // 26
            row = chr(remainder + ord('A')) + row
            if row_index == 0: break
        return row
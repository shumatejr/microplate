"""Class definition for the microplate MTP class

A Microtiter plate class for python, simplifying an internal array to the way 
people usually reference/work with microtiter plates. Currently holds a name, 
supports different densities (up to 3456), and can store arbitrary regions.

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
    ROW_LABELS : List of str
        Input row labels for microplates, based on personal experience. 
    rows : int
        Number of rows to store in the resulting microplate.
    columns : int
        Number of columns to store in the resulting microplate.
    blocks : int, optional
        Number of datablocks in the resulting microplate. Defaults to 1.
    __regions : Dict of str or List[str]
        Contains region names which are arbitrary sections of the plate and
        the wells that are in each region. Regions can be set as a list of
        individual wells ['A1','A5','B3'], or as ranges ['A2:B6']
    __arr : Numpy Matrix of floats
        Internal representation of the microplate data. Stored as a 3D Matrix
        where each slice of the matrix is a separate data block.
    """
    
    # Well labels for 3456-well plates and lower
    ROW_LABELS = [
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
        'Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF',
        'AG','AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','AS','AT',
        'AU','AV'
    ]
    
    def __init__(self, rows: int, columns: int, 
                 blocks: int = 1, name: str = "plate", 
                 input_files: List[Tuple[str, str, int, int]] = None,):
        
        self.rows = rows
        self.columns = columns
        
        self.name = name
        self.__regions = {}
        
        # If no input is specified, create a microplate pre-filled with zeros
        if input_files is None:
            self.blocks = blocks
            self.__arr = np.zeros((self.blocks, self.rows, self.columns))
        # In this instance, parse files to get contents of blocks
        else:
            # Support for multiple data blocks inside a file
            self.blocks = len(input_files)
            self.__arr = np.zeros((self.blocks, self.rows, self.columns))
            
            for block_num, data_block in enumerate(input_files):
                self.__arr[block_num] = np.array(self.__parse_file(*data_block))
        
        # Iterator defaults to none because it breaks deepcopy otherwise
        self.__index = None
        self.__iter = None
        self.__iterate = True
    
    # Internal method for parsing data from files during object construction
    def __parse_file(self, file_path: str, delimiter: str, 
                     file_row: int, file_column: int) -> NDArray:
        
        # Initialize np matrix of appropriate size to store our data
        data = np.zeros((self.rows,self.columns))
        
        # Open the file at the file path
        with open(file_path, 'r') as file:
            # Iterate through the files rows to the desired line number
            for line_number, line in enumerate(file, start=1):
                # Only consider rows within range of file_row to size of data
                if (line_number >= file_row and 
                    line_number <  file_row + self.rows):
                    
                    # Break up the line by the specified delimiter
                    split_line = line.split(delimiter)
                    
                    # Iterate through columsn to the desired column number
                    for column_number, value in enumerate(split_line, start=1):
                        if (column_number >= file_column and 
                            column_number <  file_column + self.columns):
                            
                            # Fail if non-numeric input is found in file
                            try:
                                data[line_number-file_row, 
                                        column_number-file_column] = value
                            except ValueError:
                                print("Invalid input. Check input parameters.")
                                sys.exit(1)
        return data
    
    # Internal method to convert well labels to numpy indices
    def __well_transform(self,key: str) -> Tuple[int, int]:
        # Extra up to two char letters for rows, two digit nums for columns
        row_label_arr = re.findall("[A-Z]{1,2}", key)
        col_label_arr = re.findall("[0-9]{1,2}", key)
        
        if len(row_label_arr) != 0:
            row_label = self.ROW_LABELS.index(row_label_arr[0])
        else:
            row_label = -1
        
        if len(col_label_arr) != 0:
            col_label = int(col_label_arr[0]) - 1
        else:
            col_label = -1
        
        return row_label, col_label
    
    # Return slices based on the defined regions
    # def __range_transform(self, first_well: str, second_well: str):
        # (row_1,col_1) = self.__well_transform(first_well)
        # (row_2,col_2) = self.__well_transform(second_well)
        
        # return slice(row_1, row_2+1), slice(col_1, col_2+1)
    
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
        """
        
        # Regex pattern to match anywhere from 'A1' to 'AA24:AP24'
        # Note this can still fail later if something like PA24 is passed
        regex_pattern = '^[A-Z]{1,2}\d{1,2}(:{1}[A-Z]{1,2}\d{1,2})?$'
        
        # Enforce type of str or List[str], individual str is converted to list
        if type(wells) is str:
            wells = [wells]
        if type(wells) is list and all(type(well) is str for well in wells):
            
            # Check if all the wells in the list match the pattern
            if all(re.match(regex_pattern, well) for well in wells):
                self.__regions[name] = wells
            else:
                raise TypeError
        else:
            raise TypeError
    
    def get_region(self, name: str, block: int = 1) -> NDArray:
        """Retrieve a previously set region as a 1D array.
        
        Parameters
        ----------
        name : str
            Specified name of the region which will be used for retrieval.
        block : int, optional
            Data block number to use when retrieving wells.
        
        Returns
        ------
        NDArray
            1D Numpy array of wells specified in the region.
        """
        # Check if block accessed is within the range
        if block < 1 or block > self.blocks:
            raise ValueError
        
        well_list = self.__regions[name]
        # Numpy array instead? Numpy prefers preallocated size.
        value_list = []
        
        # Parse each section and add it to the input
        for wells in well_list:
            # Well is a Range
            if re.search(':', wells):
                
                result = wells.split(":")
                (row_1,col_1) = self.__well_transform(result[0])
                (row_2,col_2) = self.__well_transform(result[1])
                
                value_list.extend(self.__arr[block-1,
                                             row_1:(row_2+1),
                                             col_1:(col_2+1)
                                            ].flatten().tolist())
            # Single well
            else:
                (row, col) = self.__well_transform(wells)
                value_list.append(self.__arr[block-1,row,col].tolist())
        return np.array(value_list)
    
    # Check if all regions passed are valid, if not raise an exception
    def _region_check(*args):
        for region in args:
            if not region in self.__regions:
                raise ValueError(f"Region {region} does not exist")
                
    # IMPROVEMENT: repeated code in normalizations, create internal method?
    def normalize_zscore(self, region_name: str = None, block: int = -1):
        """Create a new MTP that is normalized by z-score
        
        Parameters
        ----------
        region_name : str, optional
            Name of region used for normalization. If none specified, this will
            use the entire plate as the region to normalize to.
        block : int, optional
            Data block number to use when retrieving wells. If none specified,
            then all blocks of the MTP are normalized on a bock-by-block basis.
        
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
        if not type(block) is int or block > self.blocks:
            raise ValueError
        if not region_name is None and not region_name in self.__regions:
            raise ValueError
        
        norm_plate = copy.deepcopy(self)
        
        # If no block is specified, perform normalization on by-block basis
        if  block <= 0:
            block_range = range(self.blocks)
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
                    / np.std(self.get_region(region_name, block_num+1), ddof=1)
                )
        return norm_plate
    
    def normalize_percent(self, region_high: str, region_low: str, 
                          block: int = -1, method: str = 'median', 
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
        if not type(block) is int or block > self.blocks:
            raise ValueError
        if not region_high in self.__regions:
            raise ValueError
        if not region_low in self.__regions:
            raise ValueError
        
        # If no block is specified, perform normalization on by-block basis
        if  block <= 0:
            block_range = range(self.blocks)
        # Otherwise, only normalize specified block
        else:
            block_range = [block-1]
        
        norm_plate = copy.deepcopy(self)
        
        for block_num in block_range:
            if method == 'median':
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
            elif method == 'mean':
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
            elif method == 'minmax':
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
    
    # Overloaded list operations
    # Get supports specific whole plate, row, column, or specific well
    # IMPROVEMENT: There is a lot of duplicated code between get/set
    def __getitem__(self, key):
        # By default, return first data block, but if specified return block_num
        block_num = 0
        if type(key) is tuple:
            if (len(key) == 2 and isinstance(key[1], int) 
                and key[1] > 0 and key[1] <= self.blocks):
                # Well is first element of tuple, block_num second
                key, block_num = key
                block_num -= 1
            else:
                raise  ValueError("Invalid Input")
        
        # Invalid Entry
        if re.search('[^A-Z0-9:]', key):
            raise ValueError("Invalid Input")
        # Range search
        elif re.search(':', key):
            result = key.split(":")
            
            (row_1,col_1) = self.__well_transform(result[0])
            (row_2,col_2) = self.__well_transform(result[1])
            
            return self.__arr[block_num,row_1:(row_2+1),col_1:(col_2+1)]
        else:
            (row,col) = self.__well_transform(key)
            
            # Give me the whole array as 1D
            if row == -1 and col == -1: 
                return self.__arr[block_num]
            # Give me an entire row
            elif row == -1: 
                return self.__arr[block_num,:,col]
            # Give me an entire column
            elif col == -1: 
                return self.__arr[block_num,row,:]
            # Give me a well
            else: 
                return self.__arr[block_num,row,col]
    
    # Create an error_check function for the key to reduce duplication
    # Refactor to set block_num, row, col to appropriat evalue (can set to :)
    def __setitem__(self, key, value):
        # By default, return first data block, but if specified return block_num
        block_num = 0
        if type(key) is tuple:
            if (len(key) == 2 and isinstance(key[1], int) 
                and key[1] > 0 and key[1] <= self.blocks):
                # Well is first element of tuple, block_num second
                key, block_num = key
                block_num -= 1
            else:
                raise  ValueError("Invalid Input")
        
        # Invalid Entry
        if re.search('[^A-Z0-9:]', key):
            raise ValueError("Invalid Input")
        # Range search
        elif re.search(':', key):
            result = key.split(":")
            
            (row_1,col_1) = self.__well_transform(result[0])
            (row_2,col_2) = self.__well_transform(result[1])
            
            self.__arr[block_num,row_1:(row_2+1),col_1:(col_2+1)] = value
        else:
            (row,col) = self.__well_transform(key)
            
            # Give me the whole array as 1D
            if row == -1 and col == -1: 
                self.__arr[block_num] = value
            # Give me an entire row
            elif row == -1: 
                self.__arr[block_num,:,col] = value
            # Give me an entire column
            elif col == -1: 
                self.__arr[block_num,row,:] = value
            # Give me a well
            else: 
                self.__arr[block_num,row,col] = value
        
    def __delitem__(self, key):
        self.__setitem__(key, 0)
        
    # Custom string representation of a MTP
    def __str__(self):
        
        # Two decimal limit on print
        row_matrix = np.array2string(
            self.__arr + 0.0, # 0.0 is added to eliminate -0.0 from display
            formatter={'float_kind': lambda x: "%.2f" % x}, 
            max_line_width = 99999, 
            threshold=1536,
        )
        
        return f"{self.name}\n#Blocks:{self.blocks} #Rows:{self.rows} " \
               f"#Columns:{self.columns}\n{row_matrix}\n"
    
    # Iterator addition
    def index_to_well(self, row_index, column_index):
        return f"{self.ROW_LABELS[row_index]}{column_index+1}"
    # Utilize call to set up the iterator
    def __call__(self, block: int = 1):
        self.__iter = np.nditer(self.__arr[block-1], flags=['multi_index'])
        return self
    def __iter__(self):
        return self
    def __next__(self):
        if not self.__iterate:
            # Reset iterator to default values
            self.__index = None
            self.__iter = None
            self.__iterate = True
            
            raise StopIteration
        
        # iternext() automatically advances iterator, so call it last and store
        self.__index = self.__iter.multi_index
        well_value = self.__iter[0]
        self.__iterate = self.__iter.iternext()
        
        return (self.__index,self.index_to_well(*self.__index)), well_value

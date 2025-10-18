import pytest
import numpy as np
from microplate import MTP

@pytest.fixture
def plate_96():
    return MTP(
        rows = 8,
        columns = 12,
        input_files = [
            ("sample_data.txt", "\t", 23, 2),
            ("sample_data.txt", "\t", 33, 2),
        ]
    )

def test_import():
    # Test imports without any default data
    plate_default = MTP(rows = 32, columns = 48)
    assert plate_default["A1"] == 0
    assert plate_default["AF48"] == 0
    
    # Improper import from file
    with pytest.raises(ValueError):
        plate_failure = MTP(
            rows = 1,
            columns = 1,
            input_files = [("sample_data.txt",",",1,1)]
        )


def test_read(plate_96):
    
    # Simple read access
    assert plate_96["A1"] == 1.234
    assert plate_96["A12"] == 0.203
    assert plate_96["H1"] == 0.098
    assert plate_96["H12"] == 0.114
    
    # Look at different data blocks
    assert plate_96["A1",1] == 1.234
    assert plate_96["A1",2] == 0.051
    
    # Access rows/columns/blocks
    assert plate_96["B"].flatten().tolist() == [1.241,1.235,1.244,0.915,0.907,0.912,0.520,0.518,0.523,0.208,0.211,0.206]
    assert plate_96["3"].flatten().tolist() == [1.237,1.244,1.230,1.236,0.103,0.106,0.102,0.100]
    assert plate_96["",2].tolist() == (
        [[0.051,0.450,0.985,1.583,1.887,1.865,1.902,1.914,1.930,1.944,1.958,1.987],
        [0.049,0.462,0.998,1.601,1.892,1.871,1.908,1.922,1.935,1.951,1.964,2.011],
        [0.053,0.781,1.512,1.823,1.899,1.905,1.911,1.928,1.942,1.955,1.970,1.995],
        [0.052,0.769,1.531,1.845,1.911,1.918,1.924,1.933,1.948,1.960,1.975,1.989],
        [0.050,1.102,1.765,1.890,1.922,1.929,1.936,1.941,1.956,1.968,1.982,2.003],
        [0.054,1.121,1.789,1.905,1.933,1.938,1.944,1.949,1.961,1.972,1.988,1.998],
        [0.055,1.455,1.855,1.911,1.940,1.945,1.950,1.955,1.966,1.976,1.991,2.008],
        [0.056,1.448,1.861,1.924,1.949,1.952,1.958,1.962,1.971,1.980,1.996,2.001]]
    )
    
    # Access ranges
    assert plate_96["A1:B2"].flatten().tolist() == [1.234,1.229,1.241,1.235]
    assert plate_96["G11:",2].flatten().tolist() == [1.991,2.008,1.996,2.001]
    
    # Invalid access
    with pytest.raises(ValueError):
        plate_96["@"]
        plate_96["A1",3]
        plate_96["A1",-1]
    with pytest.raises(TypeError):
        plate_96[["A1","B2"]]

def test_write(plate_96):
    # Set data
    plate_96["A1"] = 1
    plate_96["A2",1] = 2
    plate_96["A3",2] = 3
    
    assert plate_96["A1"] == 1
    assert plate_96["A2"] == 2
    assert plate_96["A3",2] == 3
    
    plate_96[""] = 0
    assert plate_96[""].tolist() == (
        [[0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0]]
    )
    
    # Create a new block and check block math
    plate_96.add_block()
    
    plate_96["",3] = plate_96["",1] + plate_96["",2]
    assert (plate_96["",3] == plate_96["",2]).all()
    
    plate_96["",3] = plate_96["",1] * plate_96["",2]
    assert (plate_96["",3] == plate_96["",1]).all()
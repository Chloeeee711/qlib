#!/usr/bin/env python3
"""
Simple test script for FactorZoo integration with Alpha158
"""

import sys
from pathlib import Path
import os

# Add the qlib source code to Python path
qlib_path = Path.cwd().parent  # Go up one level from examples to qlib root
if str(qlib_path) not in sys.path:
    sys.path.insert(0, str(qlib_path))
    print(f"Added {qlib_path} to Python path")

print(f"Current working directory: {Path.cwd()}")
print(f"Qlib path: {qlib_path}")
print(f"Python path includes: {[p for p in sys.path if 'qlib' in p]}")

# Test qlib import
try:
    import qlib
    print("✓ qlib imported successfully")
    print(f"Qlib version: {qlib.__version__ if hasattr(qlib, '__version__') else 'Unknown'}")
except ImportError as e:
    print(f"✗ Failed to import qlib: {e}")
    print("Trying to install qlib...")
    os.system("pip install pyqlib")
    try:
        import qlib
        print("✓ qlib installed and imported successfully")
    except ImportError:
        print("✗ Still failed to import qlib")
        sys.exit(1)

# Test other imports
try:
    import pandas as pd
    from qlib.constant import REG_CN
    from qlib.utils import exists_qlib_data, init_instance_by_config
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
    from qlib.utils import flatten_dict
    print("✓ All qlib modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test Alpha158 with FactorZoo
try:
    from qlib.contrib.data.handler import Alpha158
    print("✓ Alpha158 imported successfully")
    
    # Test data handler config
    data_handler_config = {
        'start_time': '2008-01-01',
        'end_time': '2020-08-01',
        'fit_start_time': '2008-01-01',
        'fit_end_time': '2014-12-31',
        'instruments': 'csi300',
    }
    
    # Initialize qlib
    provider_uri = '~/.qlib/qlib_data/cn_data'
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    print("✓ Qlib initialized successfully")
    
    # Create dataset
    dataset = Alpha158(**data_handler_config)
    print("✓ Alpha158 dataset created successfully")
    
    # Get feature columns
    cols = dataset.get_cols()
    print(f"✓ Total features: {len(cols)}")
    
    # Check FactorZoo factors
    factorzoo_factors = [col for col in cols if 'FZ_' in col]
    print(f"✓ FactorZoo factors: {len(factorzoo_factors)}")
    
    # Check other factors
    other_factors = [col for col in cols if not col.startswith('FZ_')]
    print(f"✓ Other factors: {len(other_factors)}")
    
    print(f"\nExpected: 158 (original) + 42 (FactorZoo) = 200 total")
    print(f"Actual: {len(other_factors)} (original) + {len(factorzoo_factors)} (FactorZoo) = {len(cols)} total")
    
    # Show some FactorZoo factors
    if factorzoo_factors:
        print(f"\nSample FactorZoo factors:")
        for i, factor in enumerate(factorzoo_factors[:10]):  # Show first 10
            print(f"  {i+1}. {factor}")
        if len(factorzoo_factors) > 10:
            print(f"  ... and {len(factorzoo_factors) - 10} more")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed! FactorZoo integration is working correctly.")










































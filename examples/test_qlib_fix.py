#!/usr/bin/env python3
"""
Test script to verify qlib import fix
"""

import sys
import subprocess

def install_dependencies():
    """Install required dependencies"""
    dependencies = ["setuptools_scm", "pyqlib"]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                 capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"✓ {dep} installed successfully")
            else:
                print(f"⚠ {dep} install warning: {result.stderr[:200]}...")
        except Exception as e:
            print(f"⚠ {dep} install error: {e}")

def test_imports():
    """Test all required imports"""
    try:
        import qlib
        print("✓ qlib imported")
        
        import pandas as pd
        from qlib.constant import REG_CN
        from qlib.utils import exists_qlib_data, init_instance_by_config
        from qlib.workflow import R
        from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
        from qlib.utils import flatten_dict
        
        print("✓ All modules imported successfully!")
        print(f"Qlib version: {getattr(qlib, '__version__', 'Unknown')}")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing qlib import fix...")
    
    # Install dependencies
    install_dependencies()
    
    # Test imports
    if test_imports():
        print("🎉 Success! All imports working correctly.")
    else:
        print("❌ Failed! Please check your environment.")
        print("Try running: pip install setuptools_scm pyqlib")














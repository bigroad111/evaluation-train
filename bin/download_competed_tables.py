import sys
sys.path.insert(0, '../')

from valuate.db import process_tables

if __name__ == "__main__":
    """
    只存储竞品相关表.
    """
    process_tables.store_competed_tables()
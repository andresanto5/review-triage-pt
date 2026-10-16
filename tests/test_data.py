import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from train import normalize,split_temporal

class DataTests(unittest.TestCase):
    def test_normalization_dedup_key(self):
        self.assertEqual(normalize('  ÓTIMO   produto '),normalize('ótimo produto'))
    def test_temporal_disjoint(self):
        frame=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=100,tz='UTC'),'hash':[str(i) for i in range(100)],'target':[i%2 for i in range(100)]})
        a,b,c=split_temporal(frame)
        self.assertEqual(len(a)+len(b)+len(c),100)
        self.assertLess(a.date.max(),b.date.min())
        self.assertLess(b.date.max(),c.date.min())
if __name__=='__main__':unittest.main()

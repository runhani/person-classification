#!/bin/bash
python3 test.py --trained_model='./models/efficientnet_best.h5' --test_folder='./imgs'
python3 test.py --trained_model='./models/efficientnet_base.h5' --test_folder='./imgs'
python3 test.py --trained_model='./models/xception_base.h5' --test_folder='./imgs'
python3 test.py --trained_model='./models/xception_best.h5' --test_folder='./imgs'
python3 ensemble.py
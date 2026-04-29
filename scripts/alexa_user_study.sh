#!/bin/bash 

cd /home/frederic/DRAFT  

python setup_user_study.py --subsample 50 --seed 42
python setup_user_study.py --subsample 100 --seed 42
python setup_user_study.py --subsample 250 --seed 42
python setup_user_study.py --subsample 300 --seed 42
python setup_user_study.py --subsample 500 --seed 42
python setup_user_study.py --subsample 1000 --seed 42
python setup_user_study.py --subsample 2000 --seed 42
python setup_user_study.py --subsample 3000 --seed 42
python setup_user_study.py --seed 42
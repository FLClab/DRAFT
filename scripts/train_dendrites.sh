#!/bin/bash

python train_ddpm.py --dataset DendriticFActinDataset --dataset-path /home-local/Frederic/Datasets --save-folder /home-local/Frederic/baselines/DRAFT/DendriticFActin --subsample 250
python train_ddpm.py --dataset DendriticFActinDataset --dataset-path /home-local/Frederic/Datasets --save-folder /home-local/Frederic/baselines/DRAFT/DendriticFActin --subsample 500
python train_ddpm.py --dataset DendriticFActinDataset --dataset-path /home-local/Frederic/Datasets --save-folder /home-local/Frederic/baselines/DRAFT/DendriticFActin --subsample 1000
python train_ddpm.py --dataset DendriticFActinDataset --dataset-path /home-local/Frederic/Datasets --save-folder /home-local/Frederic/baselines/DRAFT/DendriticFActin --subsample 2000 
python train_ddpm.py --dataset DendriticFActinDataset --dataset-path /home-local/Frederic/Datasets --save-folder /home-local/Frederic/baselines/DRAFT/DendriticFActin --subsample 3000
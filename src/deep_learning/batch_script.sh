#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_1_e-2 --num_epochs 1 --lr 1e-2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_1_e-3 --num_epochs 1 --lr 1e-3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_1_e-4 --num_epochs 1 --lr 1e-4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_1_e-5 --num_epochs 1 --lr 1e-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_1_e-6 --num_epochs 1 --lr 1e-6

CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_2_e-2 --num_epochs 2 --lr 1e-2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_2_e-3 --num_epochs 2 --lr 1e-3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_2_e-4 --num_epochs 2 --lr 1e-4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_2_e-5 --num_epochs 2 --lr 1e-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_2_e-6 --num_epochs 2 --lr 1e-6

CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_3_e-2 --num_epochs 3 --lr 1e-2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_3_e-3 --num_epochs 3 --lr 1e-3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_3_e-4 --num_epochs 3 --lr 1e-4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_3_e-5 --num_epochs 3 --lr 1e-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_3_e-6 --num_epochs 3 --lr 1e-6

CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_5_e-2 --num_epochs 5 --lr 1e-2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_5_e-3 --num_epochs 5 --lr 1e-3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_5_e-4 --num_epochs 5 --lr 1e-4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_5_e-5 --num_epochs 5 --lr 1e-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_5_e-6 --num_epochs 5 --lr 1e-6

CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_10_e-2 --num_epochs 10 --lr 1e-2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_10_e-3 --num_epochs 10 --lr 1e-3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_10_e-4 --num_epochs 10 --lr 1e-4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_10_e-5 --num_epochs 10 --lr 1e-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_10_e-6 --num_epochs 10 --lr 1e-6

CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_20_e-2 --num_epochs 20 --lr 1e-2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_20_e-3 --num_epochs 20 --lr 1e-3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_20_e-4 --num_epochs 20 --lr 1e-4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_20_e-5 --num_epochs 20 --lr 1e-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python fast-bert.py --data_path ./data_split/ --train_file train_climate.csv --val_file val_climate.csv --label_path ./ --labels_file labels.csv --output_dir climate_20_e-6 --num_epochs 20 --lr 1e-6


#!/bin/bash
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_1_e-2 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_2_e-2 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_3_e-2 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_5_e-2 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_10_e-2 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_20_e-2 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/

CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_1_e-3 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_2_e-3 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_3_e-3 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_5_e-3 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_10_e-3 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_20_e-3 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/

CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_1_e-4 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_2_e-4 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_3_e-4 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_5_e-4 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_10_e-4 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_20_e-4 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/

CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_1_e-5 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_2_e-5 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_3_e-5 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_5_e-5 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_10_e-5 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_20_e-5 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/

CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_1_e-6 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_2_e-6 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_3_e-6 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_5_e-6 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_10_e-6 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/
CUDA_VISIBLE_DEVICES=5,6,7 python test_report.py --file_in ./data_split/feminist_test/ --model_path feminist_20_e-6 --label_path ./ --file_out_tag _prediction.txt --truth ./data_split/feminist_truth/

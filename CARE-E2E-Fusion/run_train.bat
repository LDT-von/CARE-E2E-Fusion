@echo off
setlocal
cd /d "%~dp0"
python train.py --dataset real --csv_path "blca_slides.csv" --data_root_dir "E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files" --gpu 0 --results_dir "results" --embed_dim 768 --num_tasks 1
endlocal

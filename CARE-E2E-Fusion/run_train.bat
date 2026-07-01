@echo off
pushd "c:\Users\cwnu\Desktop\[CARE-E2E-Fusion\CARE-E2E-Fusion"
D:\Anaconda3\python.exe main.py --dataset real --csv_path "c:\Users\cwnu\Desktop\[CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv" --data_root "E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files" --gpu 0 --results_dir "c:\Users\cwnu\Desktop\[CARE-E2E-Fusion\CARE-E2E-Fusion\results" --embed_dim 768 --num_tasks 1
popd

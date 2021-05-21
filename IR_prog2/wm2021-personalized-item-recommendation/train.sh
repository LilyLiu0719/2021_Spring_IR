python3 train.py --lr 0.1 \
                 --batch_size 1024 \
                 --n_factor 256 \
                 --val_ratio 0.89 \
                 --rlambda 0.005 \
                 --epochs 200 \
                 -o results/submission_1e-1_1024_256_5e-3_40_100_01.csv \
                 -m models/model_1e-1_1024_256_5e-3_40_100_01.pth \
                 -t train.csv

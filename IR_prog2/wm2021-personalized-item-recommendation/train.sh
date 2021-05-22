python3 train.py --lr 0.1 \
                 --batch_size 1024 \
                 --n_factor 256 \
                 --val_ratio 0.89 \
                 --rlambda 0.005 \
                 --epochs 200 \
                 -o results/submission_test_1-3.csv \
                 -m models/model_test_1-3.pth \
                 -t train.csv

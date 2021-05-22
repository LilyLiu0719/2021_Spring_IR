python3 train_bce.py --lr 0.1 \
                     --batch_size 512 \
                     --n_factor 128 \
                     --val_ratio 0.11\
                     --epochs 200 \
                     -o results/submission_bce_1e-1_512_128.csv \
                     -m models/model_bce_1e-1_512_128.pth \
                     -t train.csv

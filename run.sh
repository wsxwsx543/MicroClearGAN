python train.py --dataroot /home/sixuan-wu/research/MicroClear/data_splits --data_split_file train_hist.txt --name hist --model cycle_gan --n_epochs 10 --n_epochs_decay 10
python train.py --dataroot /home/sixuan-wu/research/MicroClear/data_splits --data_split_file train_rbc.txt --name rbc --model cycle_gan --n_epochs 10 --n_epochs_decay 10
python train.py --dataroot /home/sixuan-wu/research/MicroClear/data_splits --data_split_file train_focus.txt --name focus --model cycle_gan --n_epochs 50 --n_epochs_decay 50
# the ones in checkpoints directory was wrong, those were trained with RGB settings
python train.py --dataroot /home/sixuan-wu/research/MicroClear/data_splits --checkpoints_dir ./ckpt --data_split_file train_bbbc006_w1.txt --name bbbc006_w1 --model cycle_gan --n_epochs 15 --n_epochs_decay 15 --input_nc 1 --output_nc 1
python train.py --dataroot /home/sixuan-wu/research/MicroClear/data_splits --checkpoints_dir ./ckpt --data_split_file train_bbbc006_w2.txt --name bbbc006_w2 --model cycle_gan --n_epochs 15 --n_epochs_decay 15 --input_nc 1 --output_nc 1

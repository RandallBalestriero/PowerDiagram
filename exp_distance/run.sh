
for data in 0 1
do
    screen -dmS distancesaa bash -c "export CUDA_VISIBLE_DEVICES=$((data+5));
                python -i quickstart_classification.py --data_augmentation $data --dataset cifar10 --model largedense";
    screen -dmS distancesab bash -c "export CUDA_VISIBLE_DEVICES=$((data+3));
                python -i quickstart_classification.py --data_augmentation $data --dataset mnist --model largedense";
    screen -dmS distancesac bash -c "export CUDA_VISIBLE_DEVICES=$((data+1));
                python -i quickstart_classification.py --data_augmentation $data --dataset svhn --model largedense";
done

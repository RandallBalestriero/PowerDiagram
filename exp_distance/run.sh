
for data in 0 1
do
    screen -dmS distancesa bash -c "export CUDA_VISIBLE_DEVICES=$((data+5));
                python quickstart_classification.py --data_augmentation $data --dataset cifar10";
    screen -dmS distancesb bash -c "export CUDA_VISIBLE_DEVICES=$((data+3));
                python quickstart_classification.py --data_augmentation $data --dataset mnist";
    screen -dmS distancesc bash -c "export CUDA_VISIBLE_DEVICES=$((data+1));
                python quickstart_classification.py --data_augmentation $data --dataset svhn";
done

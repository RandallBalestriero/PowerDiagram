
for data in 0 1
do
#    screen -dmS distancesa bash -c "export CUDA_VISIBLE_DEVICES=7;
#                python quickstart_classification.py --data_augmentation $data --dataset cifar10";
    screen -dmS distancesb bash -c "export CUDA_VISIBLE_DEVICES=1;
                python quickstart_classification.py --data_augmentation $data --dataset mnist";
done

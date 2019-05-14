
for data in 0 1
do
    for model in cnn resnet
    do
        screen -dmS regionsa bash -c "export CUDA_VISIBLE_DEVICES=$((5+data));
                python quickstart_classification.py --model $model --data_augmentation $data --dataset mnist";
        screen -dmS regionsb bash -c "export CUDA_VISIBLE_DEVICES=$((2+data));
                python quickstart_classification.py --model $model --data_augmentation $data --dataset svhn";
        screen -dmS regionsa bash -c "export CUDA_VISIBLE_DEVICES=$((3+data));
                python quickstart_classification.py --model $model --data_augmentation $data --dataset cifar10";
    done
done

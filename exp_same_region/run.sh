
for data in 0
do
    for model in cnn resnet
    do
        screen -dmS regions bash -c "export CUDA_VISIBLE_DEVICES=$((5+data));
                python quickstart_classification.py --model $model --data_augmentation $data --dataset cifar10";
    done
done

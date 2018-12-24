#BN MODEL LR DATASET
screen -dmS A bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='0';python run_pretraining.py 0 0 0 CIFAR";
screen -dmS B bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='1';python run_pretraining.py 0 0 1 CIFAR";
screen -dmS C bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='2';python run_pretraining.py 0 0 2 CIFAR";

screen -dmS D bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='0';python run_pretraining.py 0 1 0 CIFAR";
screen -dmS E bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='1';python run_pretraining.py 0 1 1 CIFAR";
screen -dmS F bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='2';python run_pretraining.py 0 1 2 CIFAR";


screen -dmS G bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='0';python run_pretraining.py 1 0 0 CIFAR";
screen -dmS H bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='1';python run_pretraining.py 1 0 1 CIFAR";
screen -dmS I bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='2';python run_pretraining.py 1 0 2 CIFAR";

screen -dmS J bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='0';python run_pretraining.py 1 1 0 CIFAR";
screen -dmS K bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='1';python run_pretraining.py 1 1 1 CIFAR";
screen -dmS L bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='2';python run_pretraining.py 1 1 2 CIFAR";





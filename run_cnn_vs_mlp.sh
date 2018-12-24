#screen -dmS A bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='0';python run_bias_constraint.py 0 0 constrained CIFAR";
#screen -dmS B bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='1';python run_bias_constraint.py 0 0 unconstrained CIFAR";
#screen -dmS C bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='3';python run_bias_constraint.py 0 0 zero CIFAR";
screen -dmS ZZ bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='3';python run_cnn_vs_mlp.py FASHION";
screen -dmS YY bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='3';python run_cnn_vs_mlp.py CIFAR";
screen -dmS XX bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='4';python run_cnn_vs_mlp.py SVHN";
screen -dmS WW bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='6';python run_cnn_vs_mlp.py CIFAR100";



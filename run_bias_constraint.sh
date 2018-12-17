screen -dmS A bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='0';python run_bias_constraint.py 0 constrained CIFAR";
screen -dmS B bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='1';python run_bias_constraint.py 0 unconstrained CIFAR";
screen -dmS C bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='3';python run_bias_constraint.py 0 zero CIFAR";
screen -dmS D bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='4';python run_bias_constraint.py 1 constrained CIFAR";
screen -dmS E bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='5';python run_bias_constraint.py 1 unconstrained CIFAR";
screen -dmS F bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='6';python run_bias_constraint.py 1 zero CIFAR";


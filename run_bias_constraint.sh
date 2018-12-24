#screen -dmS A bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='0';python run_bias_constraint.py 0 0 constrained CIFAR";
#screen -dmS B bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='1';python run_bias_constraint.py 0 0 unconstrained CIFAR";
#screen -dmS C bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='3';python run_bias_constraint.py 0 0 zero CIFAR";
screen -dmS AD bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='3';python run_bias_constraint.py 0 1 constrained FASHION";
screen -dmS AE bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='3';python run_bias_constraint.py 0 1 unconstrained FASHION";
screen -dmS AF bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='4';python run_bias_constraint.py 0 1 zero FASHION";
screen -dmS AAD bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='6';python run_bias_constraint.py 0 0 constrained FASHION";
screen -dmS AAE bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='6';python run_bias_constraint.py 0 0 unconstrained FASHION";
screen -dmS AAF bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='6';python run_bias_constraint.py 0 0 zero FASHION";



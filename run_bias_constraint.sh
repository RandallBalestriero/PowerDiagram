screen -dmS A bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='0';python run_bias_constraint.py 0 constrained SVHN";
screen -dmS B bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='2';python run_bias_constraint.py 0 unconstrained SVHN";
screen -dmS C bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='3';python run_bias_constraint.py 0 zero SVHN";
screen -dmS D bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='4';python run_bias_constraint.py 1 constrained SVHN";
screen -dmS E bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='5';python run_bias_constraint.py 1 unconstrained SVHN";
screen -dmS F bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='7';python run_bias_constraint.py 1 zero SVHN";


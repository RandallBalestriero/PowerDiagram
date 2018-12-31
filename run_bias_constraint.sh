#screen -dmS A bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='6';python run_bias_constraint.py 0 1 explicit CIFAR";
#screen -dmS B bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='7';python run_bias_constraint.py 0 0 explicit CIFAR";
#screen -dmS C bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='6';python run_bias_constraint.py 0 1 explicit FASHION";
#screen -dmS D bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='7';python run_bias_constraint.py 0 0 explicit FASHION";
#screen -dmS E bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='3';python run_bias_constraint.py 0 1 explicit CIFAR100";
#screen -dmS F bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='4';python run_bias_constraint.py 0 0 explicit CIFAR100";
#screen -dmS G bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='3';python run_bias_constraint.py 0 1 explicit SVHN";
#screen -dmS H bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='4';python run_bias_constraint.py 0 0 explicit SVHN";


##### TO DO : constrained unconstrained 

screen -dmS AA bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='4';python run_bias_constraint.py 0 1 explicit CIFAR";
screen -dmS BB bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='4';python run_bias_constraint.py 0 0 explicit CIFAR";
screen -dmS CC bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='7';python run_bias_constraint.py 0 1 explicit FASHION";
screen -dmS DD bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='7';python run_bias_constraint.py 0 0 explicit FASHION";
screen -dmS EE bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='5';python run_bias_constraint.py 0 1 explicit CIFAR100";
screen -dmS FF bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='5';python run_bias_constraint.py 0 0 explicit CIFAR100";
screen -dmS GG bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='5';python run_bias_constraint.py 0 1 explicit SVHN";
screen -dmS HH bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='5';python run_bias_constraint.py 0 0 explicit SVHN";





#screen -dmS C bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='3';python run_bias_constraint.py 0 0 zero CIFAR";
#screen -dmS AD bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='3';python run_bias_constraint.py 0 1 constrained FASHION";
#screen -dmS AE bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='3';python run_bias_constraint.py 0 1 unconstrained FASHION";
#screen -dmS AF bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='4';python run_bias_constraint.py 0 1 zero FASHION";
#screen -dmS AAD bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='6';python run_bias_constraint.py 0 0 constrained FASHION";
#screen -dmS AAE bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='6';python run_bias_constraint.py 0 0 unconstrained FASHION";
#screen -dmS AAF bash -c "sleep 1;export CUDA_VISIBLE_DEVICES='6';python run_bias_constraint.py 0 0 zero FASHION";



python3 -u main.py --alg fedavg --dataset tinyimagenet --batch 16 --device cuda:0 --comms 120 --lr 1e-4 --model_type L2P --alpha 0.1 --instance_label 1
echo "fedavg tinyimagenet comms=120 lr=1e-4 alpha=0.1 done! (1)"
python3 -u main.py --alg fedavg --dataset tinyimagenet --batch 16 --device cuda:0 --comms 120 --lr 1e-4 --model_type L2P --alpha 0.1 --instance_label 2
echo "fedavg tinyimagenet comms=120 lr=1e-4 alpha=0.1 done! (2)"
python3 -u main.py --alg fedavg --dataset tinyimagenet --batch 16 --device cuda:0 --comms 120 --lr 1e-4 --model_type L2P --alpha 0.1 --instance_label 3
echo "fedavg tinyimagenet comms=120 lr=1e-4 alpha=0.1 done! (3)"
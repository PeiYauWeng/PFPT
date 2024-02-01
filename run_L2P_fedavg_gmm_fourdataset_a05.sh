python3 -u main.py --alg fedavg_gmm --dataset fourdataset --batch 16 --device cuda:0 --comms 120 --lr 5e-4 --n_clients 80 --model_type L2P --alpha 0.5 --reduce_sim_scalar 0.002 --instance_label 1
echo "fedavg_gmm fourdataset comms=120 lr=5e-4 alpha=0.5 done! (1)"
python3 -u main.py --alg fedavg_gmm --dataset fourdataset --batch 16 --device cuda:0 --comms 120 --lr 5e-4 --n_clients 80 --model_type L2P --alpha 0.5 --reduce_sim_scalar 0.002 --instance_label 2
echo "fedavg_gmm fourdataset comms=120 lr=5e-4 alpha=0.5 done! (2)"
python3 -u main.py --alg fedavg_gmm --dataset fourdataset --batch 16 --device cuda:0 --comms 120 --lr 5e-4 --n_clients 80 --model_type L2P --alpha 0.5 --reduce_sim_scalar 0.002 --instance_label 3
echo "fedavg_gmm fourdataset comms=120 lr=5e-4 alpha=0.5 done! (3)"
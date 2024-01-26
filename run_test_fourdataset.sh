#python3 -u main.py --alg scaffold --dataset fourdataset --batch 16 --device cuda:$1 --comms 3 --lr 1e-3 --n_clients 80 --alpha 0.1
#echo "scaffold fourdataset comms=3 lr=1e-3 alpha=0.1 done! (1)"
#python3 -u main.py --alg fedavg --dataset fourdataset --batch 16 --device cuda:$1 --comms 3 --lr 1e-4 --n_clients 80 --alpha 0.1
#echo "fedavg fourdataset comms=3 lr=5e-4 alpha=0.1 done! (1)"
python3 -u main.py --alg fedavg_gmm --dataset fourdataset --batch 16 --device cuda:$1 --comms 3 --lr 1e-4 --n_clients 80 --alpha 0.1
echo "fedavg fourdataset comms=3 lr=5e-4 alpha=0.1 done! (1)"
#python3 -u main.py --alg pfedpg --dataset fourdataset --batch 16 --device cuda:$1 --comms 3 --lr 1e-4 --n_clients 80 --alpha 0.1
#echo "pfedpg fourdataset comms=3 lr=1e-4 alpha=0.1 done! (1)"
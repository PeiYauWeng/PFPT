python3 -u main.py --alg scaffold --dataset fourdataset --batch 16 --device cuda:$1 --comms 120 --lr 1e-3 --n_clients 80 --alpha 0.1
echo "scaffold fourdataset comms=120 lr=1e-3 alpha=0.1 done! (1)"
python3 -u main.py --alg scaffold --dataset fourdataset --batch 16 --device cuda:$1 --comms 120 --lr 1e-3 --n_clients 80 --alpha 0.1
echo "scaffold fourdataset comms=120 lr=1e-3 alpha=0.1 done! (2)"
python3 -u main.py --alg scaffold --dataset fourdataset --batch 16 --device cuda:$1 --comms 120 --lr 1e-3 --n_clients 80 --alpha 0.1
echo "scaffold fourdataset comms=120 lr=1e-3 alpha=0.1 done! (3)"
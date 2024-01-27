python3 -u main.py --alg fedprox --dataset fourdataset --batch 16 --device cuda:$1 --comms 120 --lr 1e-4 --n_clients 80 --alpha 0.1
echo "fedprox fourdataset comms=120 lr=1e-4 alpha=0.1 done! (1)"
python3 -u main.py --alg fedprox --dataset fourdataset --batch 16 --device cuda:$1 --comms 120 --lr 1e-4 --n_clients 80 --alpha 0.1
echo "fedprox fourdataset comms=120 lr=1e-4 alpha=0.1 done! (2)"
python3 -u main.py --alg fedprox --dataset fourdataset --batch 16 --device cuda:$1 --comms 120 --lr 1e-4 --n_clients 80 --alpha 0.1
echo "fedprox fourdataset comms=120 lr=1e-4 alpha=0.1 done! (3)"
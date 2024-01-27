python3 -u main.py --alg pfedpg --dataset fourdataset --batch 16 --device cuda:$1 --comms 1200 --lr 5e-4 --n_clients 80 --alpha 0.5
echo "pfedpg fourdataset comms=1200 lr=5e-4 alpha=0.5 done! (1)"
python3 -u main.py --alg pfedpg --dataset fourdataset --batch 16 --device cuda:$1 --comms 1200 --lr 5e-4 --n_clients 80 --alpha 0.5
echo "pfedpg fourdataset comms=1200 lr=5e-4 alpha=0.5 done! (2)"
python3 -u main.py --alg pfedpg --dataset fourdataset --batch 16 --device cuda:$1 --comms 1200 --lr 5e-4 --n_clients 80 --alpha 0.5
echo "pfedpg fourdataset comms=1200 lr=5e-4 alpha=0.5 done! (3)"
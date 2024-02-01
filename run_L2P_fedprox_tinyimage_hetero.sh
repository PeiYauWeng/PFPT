python3 -u main.py --alg fedprox --dataset tinyimagenet --batch 16 --device cuda:0 --comms 120 --lr 1e-4 --model_type L2P --data_distribution manual_extreme_heterogeneity --n_dominated_class 20 --instance_label 1
echo "fedprox tinyimagenet comms=120 lr=1e-4 hetero done! (1)"
python3 -u main.py --alg fedprox --dataset tinyimagenet --batch 16 --device cuda:0 --comms 120 --lr 1e-4 --model_type L2P --data_distribution manual_extreme_heterogeneity --n_dominated_class 20 --instance_label 2
echo "fedprox tinyimagenet comms=120 lr=1e-4 hetero done! (2)"
python3 -u main.py --alg fedprox --dataset tinyimagenet --batch 16 --device cuda:0 --comms 120 --lr 1e-4 --model_type L2P --data_distribution manual_extreme_heterogeneity --n_dominated_class 20 --instance_label 3
echo "fedprox tinyimagenet comms=120 lr=1e-4 hetero done! (3)"
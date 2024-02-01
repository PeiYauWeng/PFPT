import os
import argparse
import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
import torchvision
from tqdm._tqdm import trange
import pytz
from datetime import datetime
from Algo.get_algos import get_algorithm
from data_util.DataTransformBuilder import build_data_transform
from data_util.DataDistributer import DataPartitioner
from data_util.TinyImageNet_reader import TinyImageNet_reader
from data_util.PACS_reader import Pacs_reader
from data_util.Four_dataset_reader import four_dataset_reader
from Scenario import FL_scenario
from util.TOdevice import to_device
from Models.Prompted_models import Prompted_ViT_B32
from Models.L2P_heuristic_model import L2P_ViT_B32
from Models.pFedPG_model import client_prompted_vit_b32, BaseHeadsForLocal, prompt_generator
from util.train_eval import evaluate_pFedPG, train_eval_pFedPG, evaluate_all_pFedPG, evaluate_all_pFedPG_mask
from util.saving_tools import save_eval_npy_file, save_model_trainable_part, save_pfedpg_baseHeads
os.environ['TORCH_HOME'] = '/home/qh1002/Code/Probabilistic_Prompt_Aggregation_ver2/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='Algorithm to choose: [fedavg | fedprox | fedopt | scaffold | fedavg_gmm | pfedpg | fedavg_nonpara]')
    #parser.add_argument('--scenario', type=str, default='cross_devices',
                        #help='Federated Learning Scenario to choose: [cross_devices | cross_silo]')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset to choose: [cifar10 | cifar100 | PACS | tinyimagenet | fourdataset]')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='[cuda | cpu]')
    parser.add_argument('--batch', type=int, default=64, 
                        help='batch size')
    parser.add_argument('--comms', type=int, default=100, 
                        help='communication rounds')
    parser.add_argument('--local_eps', type=int, default=5,
                        help='number of epochs in local clients training')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--n_clients', type=int, default=100,
                        help='the number of clients')
    parser.add_argument('--n_sampled_clients', type=int, default=10,
                        help='the number of sampled clients per round')
    parser.add_argument('--data_distribution', type=str, default='non_iid_dirichlet',
                        help='data split way to choose: [non_iid_dirichlet | manual_extreme_heterogeneity]')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='the level of non-iid data split')
    parser.add_argument('--n_dominated_class', type=int, default=1,
                        help='number of dominated class when applying manual_heterogeneity')
    parser.add_argument('--model_type', type=str, default='prompted',
                        help='choose model type you want: [prompted | L2P]')
    parser.add_argument('--prompt_method', type=str, default='shallow',
                        help='[shallow | deep]')
    parser.add_argument('--n_tokens', type=int, default=10,
                        help='number of tokens in prompt')
    
    # algorithm-specific parameters
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for fedprox')
    parser.add_argument('--fedopt_global_lr', type=float, default=5e-5,
                        help='Global model updating learning rate for fedopt')
    parser.add_argument('--prompt_lr', type=float, default=1e-3,
                        help='Prompt generaotr updating learning rate for pFedPG')
    parser.add_argument('--pool_size', type=int, default=20,
                        help='Define the prompt pool size in L2P model')
    parser.add_argument('--batchwise_prompt', type=bool, default=True,
                        help='Define L2P heuristic model if selecting top_k prompts in pool by batchwise')
    parser.add_argument('--nonpara_hidden', type=int, default=128,
                        help='Define the number of hidden neurons in Nonparametric aggregation method')
    parser.add_argument('--reduce_sim_scalar', type=float, default=0.01,
                        help='control reduce similarity in L2P model')
    parser.add_argument('--save_model', type=bool, default=False,
                        help='Save the trained model in the last epoch')
    parser.add_argument('--instance_label', type=int, default=0,
                        help='instance label')
    args = parser.parse_args()
    print(args.device)
    print(torch.cuda.get_device_name())
    #check if cross_devices or corss_silo
    if args.n_clients > args.n_sampled_clients:
        scenrio_type = 'cross_devices'
    else:
        scenrio_type = 'cross_silo'
    
    norm_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    resize = 256
    centercrop_size = 224
    preprocess = build_data_transform(norm_stats, resize, centercrop_size)
    class_mask = None
    
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='./Dataset', train=True, download=True, transform=preprocess
        )
        testset = torchvision.datasets.CIFAR10(
            root='./Dataset', train=False, download=True, transform=preprocess
        )
        num_classes = 10
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root='./Dataset', train=True, download=True, transform=preprocess
        )
        testset = torchvision.datasets.CIFAR100(
            root='./Dataset', train=False, download=True, transform=preprocess
        )
        num_classes = 100
    elif args.dataset == 'PACS':
        data_name = ['art_painting', 'cartoon', 'photo', 'sketch']
        trainset = [Pacs_reader('./Dataset/PACS/PACS', split=data_name[i], train=True, transform=preprocess, random_seed=i) for i in range(len(data_name))]
        testset = [Pacs_reader('./Dataset/PACS/PACS', split=data_name[i], train=False, transform=preprocess, random_seed=i) for i in range(len(data_name))]
        testset = Data.ConcatDataset(testset)
        num_classes = 7
    elif args.dataset == 'tinyimagenet':
        trainset = TinyImageNet_reader('../Probabilistic_Prompt_Aggregation_ver2/Dataset/tiny-imagenet-200/', train=True, transform=preprocess)
        testset = TinyImageNet_reader('../Probabilistic_Prompt_Aggregation_ver2/Dataset/tiny-imagenet-200/', train=False, transform=preprocess)
        num_classes = 200
    elif args.dataset == 'fourdataset':
        data_name = ['mnistm', 'fashion', 'cinic10', 'mmafedb']
        trainset, class_mask = four_dataset_reader([30000]*4, train=True, transform=preprocess)
        testset, _ = four_dataset_reader([2500]*4, train=False, transform=preprocess)
        num_classes = 37
        print(args.dataset)
    else:
        raise ValueError("Input dataset is not supported")
    print(class_mask)
    # setup global testing set & distributed dataset
    if args.dataset == 'fourdataset':
        testloader = [Data.DataLoader(testset[i], batch_size=args.batch, num_workers=2, shuffle=False) for i in range(len(testset))]
    else:
        testloader = Data.DataLoader(testset, batch_size=args.batch, num_workers=2, shuffle=False)
    if args.dataset == 'PACS' or args.dataset == 'fourdataset':
        data_partitioner = [DataPartitioner(trainset[i], args.n_clients//len(data_name)) for i in range(len(trainset))]
    else:
        data_partitioner = DataPartitioner(trainset, args.n_clients)
        
    if args.alpha == 0.1:
        seed = 871
    elif args.alpha == 0.2:
        seed = 459
    elif args.alpha == 0.3:
        seed = 429
    elif args.alpha == 0.4:
        seed = 3760
    elif args.alpha == 0.5:
        seed = 448
        
    #Get current time
    LATz = pytz.timezone("America/Los_Angeles") 
    timeInLA = datetime.now(LATz)
    
    if args.data_distribution == 'non_iid_dirichlet':
        if isinstance(data_partitioner, list):
            for i in range(len(data_partitioner)):
                if i==0:
                    data_partitioner[i].dirichlet_split_noniid(alpha=args.alpha, least_samples=32, manual_seed=seed)
                else:
                    data_partitioner[i].dirichlet_split_noniid(alpha=args.alpha, least_samples=32, manual_seed=data_partitioner[0].seed)
        else:
            data_partitioner.dirichlet_split_noniid(alpha=args.alpha, least_samples=32, manual_seed=seed)
        #setup log file for recording
        npy_save_path = f"./output_record/loss_acc/info_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_alpha{args.alpha}_({args.instance_label})"
        pool_save_path = f"./output_record/loss_acc/pool_record_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_alpha{args.alpha}_({args.instance_label})"
        log_file_local_training = open(f"./output_record/log_output/local_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_alpha{args.alpha}_({args.instance_label}).txt", mode="w+", encoding="utf-8")
        log_file_global_aggregation = open(f"./output_record/log_output/global_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_alpha{args.alpha}_({args.instance_label}).txt", 
                                           mode="a+", encoding="utf-8")
    elif args.data_distribution == 'manual_extreme_heterogeneity':
        if isinstance(data_partitioner, list):
            for i in range(len(data_partitioner)):
                data_partitioner[i].manual_allocating_noniid(args.n_dominated_class, 0.99, 1.0)
        else:
            data_partitioner.manual_allocating_noniid(args.n_dominated_class, 0.99, 1.0)
        #setup log file for recording
        npy_save_path = f"./output_record/loss_acc/info_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_({args.instance_label})"
        pool_save_path = f"./output_record/loss_acc/pool_record_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_({args.instance_label})"
        log_file_local_training = open(f"./output_record/log_output/local_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_({args.instance_label}).txt", mode="w+", encoding="utf-8")
        log_file_global_aggregation = open(f"./output_record/log_output/global_{args.model_type}_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_({args.instance_label}).txt", 
                                           mode="a+", encoding="utf-8")
    else:
        raise ValueError("Input data distribution is not supported")
    
    print("-----------------------------------------------------------------------------------------", file=log_file_global_aggregation)
    print(f"*****Current time is {timeInLA} ***** Total communication rounds are {args.comms} *****", file=log_file_global_aggregation)
    print("-----------------------------------------------------------------------------------------", file=log_file_global_aggregation)
    print(f"Learning Rate: {args.lr}; Batch size: {args.batch}", file=log_file_global_aggregation)
    #prepare to record eval loss and accuracy
    eval_loss_record = list()
    eval_acc_record = list()
    eval_pool_size = None
    eval_pool = None
    # call FL algorithm and run
    if args.alg in ['fedavg', 'fedprox', 'fedopt', 'scaffold', 'fedavg_gmm', 'fedavg_nonpara', 'fedprox_nonpara']:
        print(f"*****Current Federated Learning Scenario is {scenrio_type}*****", file=log_file_global_aggregation)
        # setup federated scenario
        if args.dataset == 'PACS' or args.dataset == 'fourdataset':
            all_clients_weights = np.concatenate([data_partitioner[i].get_all_client_weights() for i in range(len(data_partitioner))])
            distributed_dataloaders = [data_partitioner[i].get_distributed_data(batch_size=args.batch) for i in range(len(data_partitioner))]
            distributed_dataloaders = [distributed_dataloaders[i][j] for i in range(len(distributed_dataloaders)) for j in range(data_partitioner[i].n_clients)]
            print(len(distributed_dataloaders))
            fl_scen = FL_scenario(all_clients_weights,
                                  args.n_clients, args.n_sampled_clients,
                                  distributed_dataloaders)
        else:
            fl_scen = FL_scenario(data_partitioner.get_all_client_weights(),
                                  args.n_clients, args.n_sampled_clients,
                                  data_partitioner.get_distributed_data(batch_size=args.batch))
        # construct model
        print(args.model_type == 'L2P')
        if args.model_type == 'prompted':
            weight_init = 'random'
            server_model = to_device(Prompted_ViT_B32(weight_init=weight_init, 
                                                      prompt_method=args.prompt_method, 
                                                      num_tokens=args.n_tokens, 
                                                      num_classes=num_classes), args.device)
        elif args.model_type == 'L2P':
            server_model = to_device(L2P_ViT_B32(prompt_method=args.prompt_method,
                                                 batchwise_prompt=args.batchwise_prompt,
                                                 pool_size=args.pool_size,
                                                 top_k=args.n_tokens,
                                                 num_classes=num_classes), args.device)
            eval_pool_size = list()
            eval_pool = list()
            eval_pool_size.append(args.pool_size)
            eval_pool.append(server_model.pool.prompt.data.detach().cpu().numpy())
        else:
            raise ValueError("Model type is not supported")
        server_model.build_trainable_keys()
        if args.alg == 'scaffold':
            server_model.init_contorl_parameter_for_scaffold(device=args.device)
        #get FL algorithm
        if '_' in args.alg:
            associate_algo = args.alg.split('_')[0]
            algclass = get_algorithm(associate_algo)
        else:
            algclass = get_algorithm(args.alg)
        if args.alg == 'fedprox':
            algo = algclass(server_model=server_model,scenario=fl_scen,
                        loss_fun=nn.CrossEntropyLoss(), mu=args.mu, class_mask=class_mask, device=args.device)
        elif args.alg == 'fedopt':
            algo = algclass(server_model=server_model,scenario=fl_scen,
                        loss_fun=nn.CrossEntropyLoss(), global_lr=args.fedopt_global_lr, class_mask=class_mask, device=args.device)
        elif args.alg == 'fedavg_gmm' or args.alg == 'fedprox_gmm':
            print('gmm')
            if associate_algo == 'fedavg':
                algo = algclass(server_model=server_model,scenario=fl_scen,
                                loss_fun=nn.CrossEntropyLoss(), fed_method='simple_gmm_prompt', class_mask=class_mask, device=args.device)
            elif associate_algo == 'fedprox':
                algo = algclass(server_model=server_model,scenario=fl_scen,
                                loss_fun=nn.CrossEntropyLoss(), mu=args.mu, fed_method='simple_gmm_prompt', class_mask=class_mask, device=args.device)
            else:
                raise ValueError("Algorithm is not supported")
        elif args.alg == 'fedavg_nonpara' or args.alg == 'fedprox_nonpara':
            print('nonpara')
            if associate_algo == 'fedavg':
                algo = algclass(server_model=server_model,scenario=fl_scen,
                                loss_fun=nn.CrossEntropyLoss(),fed_method='nonparametric_aggregation',
                                nonpara_hidden=args.nonpara_hidden, class_mask=class_mask, device=args.device)
            elif associate_algo == 'fedprox':
                algo = algclass(server_model=server_model,scenario=fl_scen,
                                loss_fun=nn.CrossEntropyLoss(), mu=args.mu, fed_method='nonparametric_aggregation',
                                nonpara_hidden=args.nonpara_hidden, class_mask=class_mask, device=args.device)
            else:
                raise ValueError("Algorithm is not supported")
        else:
            algo = algclass(server_model=server_model,scenario=fl_scen,
                            loss_fun=nn.CrossEntropyLoss(), class_mask=class_mask, device=args.device)
        
        for comm_round in trange(args.comms):
            algo.client_train(comm_round=comm_round, epochs=args.local_eps, lr=args.lr, 
                              output_file=log_file_local_training, reduce_sim_scalar=args.reduce_sim_scalar, print_output=True)
            algo.server_aggre()
            print(f'--------------------------Round {comm_round+1} complete----------------------------',
                  file=log_file_local_training)
            if args.model_type == 'L2P':
                eval_loss, eval_acc, current_pool_size = algo.server_eval(testloader, comm_round, log_file_global_aggregation)
                eval_pool_size.append(current_pool_size)
                if (comm_round+1)%10 == 0:
                    eval_pool.append(algo.server_model.pool.prompt.data.detach().cpu().numpy())
            else:
                eval_loss, eval_acc = algo.server_eval(testloader, comm_round, log_file_global_aggregation)
            eval_loss_record.append(eval_loss)
            eval_acc_record.append(eval_acc)
        min_index = np.argmin(np.array(eval_loss_record))
        print(f'*****Final accuracy with minimal eval_loss: {eval_acc_record[min_index]}*****', 
              file=log_file_global_aggregation)
        
    elif args.alg == 'pfedpg':
        if args.dataset == 'PACS' or args.dataset == 'fourdataset':
            trainloader = [data_partitioner[i].get_distributed_data(batch_size=args.batch) for i in range(len(data_partitioner))]
            trainloader = [trainloader[i][j] for i in range(len(trainloader)) for j in range(data_partitioner[i].n_clients)]
            print(len(trainloader))
        else:
            trainloader = data_partitioner.get_distributed_data(batch_size=args.batch)
        # construct model
        clients = BaseHeadsForLocal(dataloaders=trainloader, num_classes=num_classes, local_lr=args.lr, device=args.device)
        prompt_gen = to_device(prompt_generator(num_tokens=args.n_tokens, 
                                                num_clients=args.n_clients, 
                                                k_dim=512, v_dim=512), args.device)
        vit_net = to_device(client_prompted_vit_b32(num_tokens=args.n_tokens), args.device)
        vit_net.build_trainable_keys()
        # setup distributed testset for personalized models
        if args.dataset == 'PACS' or args.dataset == 'fourdataset':
            test_partitioner = [DataPartitioner(testset[i], args.n_clients//len(data_name)) for i in range(len(testset))]
        else:
            test_partitioner = DataPartitioner(testset, args.n_clients)
        if args.data_distribution == 'non_iid_dirichlet':
            if isinstance(data_partitioner, list):
                for i in range(len(test_partitioner)):
                    if i==0:
                        test_partitioner[i].dirichlet_split_noniid(alpha=args.alpha, least_samples=1, manual_seed=seed)
                    else:
                        test_partitioner[i].dirichlet_split_noniid(alpha=args.alpha, least_samples=1, manual_seed=test_partitioner[0].seed)
            else:
                test_partitioner.dirichlet_split_noniid(args.alpha, least_samples=1, manual_seed=seed)
        elif args.data_distribution == 'manual_extreme_heterogeneity':
            if isinstance(test_partitioner, list):
                for i in range(len(test_partitioner)):
                    test_partitioner[i].manual_allocating_noniid(args.n_dominated_class, 0.99, 1.0)
            else:
                test_partitioner.manual_allocating_noniid(args.n_dominated_class, 0.99, 1.0)
        else:
            raise ValueError("Input data distribution is not supported")
        if args.dataset == 'PACS' or args.dataset == 'fourdataset':
            distributed_testloaders = [test_partitioner[i].get_distributed_data(batch_size=args.batch) for i in range(len(test_partitioner))]
            distributed_testloaders = [distributed_testloaders[i][j] for i in range(len(distributed_testloaders)) for j in range(test_partitioner[i].n_clients)]
        else:
            distributed_testloaders = test_partitioner.get_distributed_data(batch_size=args.batch)
        eval_loss_record, eval_acc_record = train_eval_pFedPG(clients=clients,
                                                              prompt_gen=prompt_gen,
                                                              vit_net=vit_net,
                                                              comm_rounds=args.comms,
                                                              local_epochs=args.local_eps,
                                                              test_loaders=distributed_testloaders,
                                                              output_file=log_file_local_training,
                                                              inner_lr=args.lr,
                                                              prompt_lr=args.prompt_lr,
                                                              print_output=True,
                                                              device=args.device,
                                                              class_mask=class_mask)
        if class_mask is not None:
            print(len(testloader), file=log_file_global_aggregation)
            evaluate_all_pFedPG_mask(clients=clients, all_test_loader=testloader, prompt_gen=prompt_gen, vit_net=vit_net, 
                                output_file=log_file_global_aggregation, device=args.device, class_mask=class_mask)
        else:
            evaluate_all_pFedPG(clients=clients, all_test_loader=testloader, prompt_gen=prompt_gen, vit_net=vit_net, 
                                output_file=log_file_global_aggregation, device=args.device)
    else:
        raise ValueError("Algorithm is not supported")
    #save record to npy files
    save_eval_npy_file(eval_loss_record=eval_loss_record, eval_acc_record=eval_acc_record, 
                       eval_pool_size=eval_pool_size, file_path=npy_save_path+'.npy')
    if args.model_type == 'L2P':
        np.save(pool_save_path+'.npy', np.concatenate(eval_pool, axis=0))
    #close txt files
    log_file_local_training.close()
    log_file_global_aggregation.close()
    
    
    
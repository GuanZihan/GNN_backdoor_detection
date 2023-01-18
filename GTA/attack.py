import sys, os
sys.path.append("../src/config_subgraph.py")
sys.path.append("..")
from utils.S2VGraph import S2VGraph
import copy
import numpy as np
from tqdm import tqdm
import torch
import networkx as nx

from utils.datareader import DataReader
from utils.bkdcdd import select_cdd_graphs, select_cdd_nodes
from utils.mask import gen_mask, recover_mask
import benig as benign
import trojan.GTA as gta
from trojan.input import gen_input
from trojan.prop import train_model, evaluate
from config import parse_args

from src.models import GCN_N

class GraphBackdoor:
    def __init__(self, args) -> None:
        self.args = args
        assert torch.cuda.is_available(), 'no GPU available'
        self.cpu = torch.device('cpu')
        self.cuda = torch.device('cuda:'+args.device)

    def run(self):
        # train a benign GNN
        # self.args.train_epochs = 1
        # self.args.gtn_epochs = 1
        # self.args.resample_steps = 1
        # self.args.bilevel_steps = 1
        self.benign_dr, self.benign_model = benign.run(self.args)
        print("train benign models")
        model = copy.deepcopy(self.benign_model).to(self.cuda)


        # pick up initial candidates
        bkd_gids_test, bkd_nids_test, bkd_nid_groups_test = self.bkd_cdd('test')

        nodenums = [adj.shape[0] for adj in self.benign_dr.data['adj_list']]
        nodemax = max(nodenums)
        featdim = np.array(self.benign_dr.data['features'][0]).shape[1]
        
        # init two generators for topo/feat
        toponet = gta.GraphTrojanNet(nodemax, self.args.gtn_layernum)
        # print(toponet)
        featnet = gta.GraphTrojanNet(featdim, self.args.gtn_layernum)

        
        # init test data
        # NOTE: for data that can only add perturbation on features, only init the topo value
        init_dr_test = self.init_trigger(
            self.args, copy.deepcopy(self.benign_dr), bkd_gids_test, bkd_nid_groups_test, 0.0, 0.0)
        bkd_dr_test = copy.deepcopy(init_dr_test)

        topomask_test, featmask_test = gen_mask(
            init_dr_test, bkd_gids_test, bkd_nid_groups_test)
        Ainput_test, Xinput_test = gen_input(self.args, init_dr_test, bkd_gids_test)

        bkd_dr_train = None
        
        for rs_step in range(self.args.resample_steps):   # for each step, choose different sample
            
            # randomly select new graph backdoor samples
            bkd_gids_train, bkd_nids_train, bkd_nid_groups_train = self.bkd_cdd('train')

            # positive/negtive sample set
            pset = bkd_gids_train
            nset = list(set(self.benign_dr.data['splits']['train'])-set(pset))

            if self.args.pn_rate != None:
                if len(pset) > len(nset):
                    repeat = int(np.ceil(len(pset)/(len(nset)*self.args.pn_rate)))
                    nset = list(nset) * repeat
                else:
                    repeat = int(np.ceil((len(nset)*self.args.pn_rate)/len(pset)))
                    pset = list(pset) * repeat
            
            # init train data
            # NOTE: for data that can only add perturbation on features, only init the topo value
            init_dr_train = self.init_trigger(
                self.args, copy.deepcopy(self.benign_dr), bkd_gids_train, bkd_nid_groups_train, 0.0, 0.0)
            bkd_dr_train = copy.deepcopy(init_dr_train)

            topomask_train, featmask_train = gen_mask(
                init_dr_train, bkd_gids_train, bkd_nid_groups_train)
            Ainput_train, Xinput_train = gen_input(self.args, init_dr_train, bkd_gids_train)
            
            for bi_step in range(self.args.bilevel_steps):
                print("Resampling step %d, bi-level optimization step %d" % (rs_step, bi_step))
                
                toponet, featnet = gta.train_gtn(
                    self.args, model, toponet, featnet,
                    pset, nset, topomask_train, featmask_train, 
                    init_dr_train, bkd_dr_train, Ainput_train, Xinput_train)
                
                # get new backdoor datareader for training based on well-trained generators
                for gid in bkd_gids_train:
                    rst_bkdA = toponet(
                        Ainput_train[gid], topomask_train[gid], self.args.topo_thrd, 
                        self.cpu, self.args.topo_activation, 'topo')
                    # rst_bkdA = recover_mask(nodenums[gid], topomask_train[gid], 'topo')
                    # bkd_dr_train.data['adj_list'][gid] = torch.add(rst_bkdA, init_dr_train.data['adj_list'][gid])
                    bkd_dr_train.data['adj_list'][gid] = torch.add(
                        rst_bkdA[:nodenums[gid], :nodenums[gid]].detach().cpu(), 
                        init_dr_train.data['adj_list'][gid])
                
                    rst_bkdX = featnet(
                        Xinput_train[gid], featmask_train[gid], self.args.feat_thrd, 
                        self.cpu, self.args.feat_activation, 'feat')
                    # rst_bkdX = recover_mask(nodenums[gid], featmask_train[gid], 'feat')
                    # bkd_dr_train.data['features'][gid] = torch.add(rst_bkdX, init_dr_train.data['features'][gid]) 
                    bkd_dr_train.data['features'][gid] = torch.add(
                        rst_bkdX[:nodenums[gid]].detach().cpu(), init_dr_train.data['features'][gid]) 
                    
                # train GNN
                train_model(self.args, bkd_dr_train, model, list(set(pset)), list(set(nset)))
                
                #----------------- Evaluation -----------------#
                for gid in bkd_gids_test:
                    rst_bkdA = toponet(
                        Ainput_test[gid], topomask_test[gid], self.args.topo_thrd, 
                        self.cpu, self.args.topo_activation, 'topo')
                    # rst_bkdA = recover_mask(nodenums[gid], topomask_test[gid], 'topo')
                    # bkd_dr_test.data['adj_list'][gid] = torch.add(rst_bkdA, 
                    #     torch.as_tensor(copy.deepcopy(init_dr_test.data['adj_list'][gid])))
                    bkd_dr_test.data['adj_list'][gid] = torch.add(
                        rst_bkdA[:nodenums[gid], :nodenums[gid]], 
                        torch.as_tensor(copy.deepcopy(init_dr_test.data['adj_list'][gid])))
                
                    rst_bkdX = featnet(
                        Xinput_test[gid], featmask_test[gid], self.args.feat_thrd, 
                        self.cpu, self.args.feat_activation, 'feat')
                    # rst_bkdX = recover_mask(nodenums[gid], featmask_test[gid], 'feat')
                    # bkd_dr_test.data['features'][gid] = torch.add(
                    #     rst_bkdX, torch.as_tensor(copy.deepcopy(init_dr_test.data['features'][gid])))
                    bkd_dr_test.data['features'][gid] = torch.add(
                        rst_bkdX[:nodenums[gid]], torch.as_tensor(copy.deepcopy(init_dr_test.data['features'][gid])))
                    
                # graph originally in target label
                yt_gids = [gid for gid in bkd_gids_test 
                        if self.benign_dr.data['labels'][gid]==self.args.target_class] 
                # graph originally notin target label
                yx_gids = list(set(bkd_gids_test) - set(yt_gids))
                clean_graphs_test = list(set(self.benign_dr.data['splits']['test'])-set(bkd_gids_test))

                # feed into GNN, test success rate
                bkd_acc = evaluate(self.args, bkd_dr_test, model, bkd_gids_test)
                flip_rate = evaluate(self.args, bkd_dr_test, model,yx_gids)
                clean_acc = evaluate(self.args, bkd_dr_test, model, clean_graphs_test)
                print(bkd_acc)
                print(clean_acc)

                
                # # save gnn
                # if rs_step == 0 and (bi_step==self.args.bilevel_steps-1 or abs(bkd_acc-100) <1e-4):
                #     if self.args.save_bkd_model:
                #         save_path = self.args.bkd_model_save_path
                #         os.makedirs(save_path, exist_ok=True)
                #         save_path = os.path.join(save_path, '%s-%s-%f.t7' % (
                #             self.args.models, self.args.dataset, self.args.train_ratio,
                #             self.args.bkd_gratio_trainset, self.args.bkd_num_pergraph, self.args.bkd_size))
                #
                #         torch.save({'models': models.state_dict(),
                #                     'asr': bkd_acc,
                #                     'flip_rate': flip_rate,
                #                     'clean_acc': clean_acc,
                #                 }, save_path)
                #         print("Trojaning models is saved at: ", save_path)
                    
                if abs(bkd_acc-100) <1e-4:
                    # bkd_dr_tosave = copy.deepcopy(bkd_dr_test)
                    print("Early Termination for 100% Attack Rate")
                    break
        print('Done')
        adj_list = bkd_dr_train.data["adj_list"]
        labels = bkd_dr_train.data["labels"]
        features = bkd_dr_train.data["features"]


        ################
        graphs = []
        backdoor_train_graphs = []
        clean_train_graphs = []
        ctx = 0
        perm_index = []
        for idx, adj in enumerate(adj_list):
            if idx in bkd_dr_train.data["splits"]["train"]:
                g = np.array(adj)
                x = features[idx]
                graph = (g, x, labels[idx])
                if idx in bkd_gids_train:
                    perm_index.append(ctx)
                    backdoor_train_graphs.append(graph)
                else:
                    clean_train_graphs.append(graph)
                ctx += 1


                graphs.append(graph)
        np.save("GTA_train_dataset_{}_{}.npy".format(args.dataset, args.bkd_size), np.array(graphs))
        np.save("GTA_train_dataset_backdoor_{}_{}.npy".format(args.dataset, args.bkd_size), np.array(backdoor_train_graphs))
        np.save("GTA_train_dataset_clean_{}_{}.npy".format(args.dataset, args.bkd_size), np.array(clean_train_graphs))
        print(len(graphs))
        print("save done!")
        print(len(adj_list), len(bkd_dr_test.data["adj_list"]))
        print(bkd_dr_train.data["splits"]["train"])
        print(bkd_gids_train)

        dj_list = bkd_dr_test.data["adj_list"]
        labels = bkd_dr_test.data["labels"]
        features = bkd_dr_test.data["features"]

        ################
        graphs = []
        for idx, adj in enumerate(dj_list):
            if idx in bkd_gids_test:
                g = np.array(adj.detach().numpy())
                x = features[idx]
                graph = (g, x.detach().numpy(), labels[idx])
                graphs.append(graph)
        print(len(graphs), graphs[0][0].shape)
        np.save("GTA_backdoor_test_dataset_{}_{}.npy".format(args.dataset, args.bkd_size), np.array(graphs))
        print("save done!")

        np.save("GTA_perm_index_{}_{}.npy".format(args.dataset, args.bkd_size), np.array(perm_index))

        print(len(bkd_gids_test))

        np.save("GTA_perm_index_test_{}_{}.npy".format(args.dataset, args.bkd_size), np.array(bkd_gids_test))
        print("save done!")

        torch.save(model.state_dict(), "gcn_{}.pt".format(args.dataset))


    def bkd_cdd(self, subset: str):
        # - subset: 'train', 'test'
        # find graphs to add trigger (not modify now)
        bkd_gids = select_cdd_graphs(
            self.args, self.benign_dr.data['splits'][subset], self.benign_dr.data['adj_list'], subset)
        # find trigger nodes per graph
        # same sequence with selected backdoored graphs
        bkd_nids, bkd_nid_groups = select_cdd_nodes(
            self.args, bkd_gids, self.benign_dr.data['adj_list'])

        assert len(bkd_gids)==len(bkd_nids)==len(bkd_nid_groups)

        return bkd_gids, bkd_nids, bkd_nid_groups


    @staticmethod
    def init_trigger(args, dr: DataReader, bkd_gids: list, bkd_nid_groups: list, init_edge: float, init_feat: float):
        if init_feat == None:
            init_feat = - 1
            print('init feat == None, transferred into -1')
        
        # (in place) datareader trigger injection
        for i in tqdm(range(len(bkd_gids)), desc="initializing trigger..."):
            gid = bkd_gids[i]           
            for group in bkd_nid_groups[i] :
                # change adj in-place
                src, dst = [], []
                for v1 in group:
                    for v2 in group:
                        if v1!=v2:
                            src.append(v1)
                            dst.append(v2)
                a = np.array(dr.data['adj_list'][gid])
                a[src, dst] = init_edge
                dr.data['adj_list'][gid] = a.tolist()

                # change features in-place
                featdim = len(dr.data['features'][0][0])
                a = np.array(dr.data['features'][gid])
                a[group] = np.ones((len(group), featdim)) * init_feat
                dr.data['features'][gid] = a.tolist()
                
            # change graph labels
            assert args.target_class is not None
            dr.data['labels'][gid] = args.target_class

        return dr  

if __name__ == '__main__':
    args = parse_args()
    attack = GraphBackdoor(args)
    attack.run()
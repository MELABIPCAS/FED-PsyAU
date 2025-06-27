import copy
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from client import train
from models.au_graph_mer import au_graph_mer


class FedProx:
    def __init__(self, args):
        self.args = args
        self.nn = au_graph_mer('au_graph_mer_softmax', args.device, 'server', out_channels=3)
        self.nns = []
        self.acc_list = [[], [], []]
        self.uf1_list = [[], [], []]
        self.uar_list = [[], [], []]
        self.train_loss_list = [[], [], []]
        self.val_loss_list = [[], [], []]
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.args.clients[i]
            self.nns.append(temp)
    def server(self):
        for t in tqdm(range(self.args.r)):
            print('--------------round', t + 1, '--------------')
            m = np.max([int(self.args.C * self.args.K), 1])
            index = random.sample(range(0, self.args.K), m)
            if t > 0:
                self.dispatch(index)
            self.client_update(index)
        print('Finished! Save all result file!')
        self.save_results_to_csv()
        return self.nn
    def dispatch(self, index):
        personalized_models = []

        for j in index:
            client_model = self.nns[j]
            other_clients_models = [self.nns[i] for i in index if i != j]
            personalized_model = self.generate_personalized_model(client_model, other_clients_models)

            personalized_models.append(personalized_model)

        for j, personalized_model in zip(index, personalized_models):
            for old_params, new_params in zip(self.nns[j].parameters(), personalized_model.parameters()):
                old_params.data = new_params.data.clone()


    # def server(self):
    #     for t in tqdm(range(self.args.r)):
    #         print('--------------round', t + 1, '--------------')
    #         m = np.max([int(self.args.C * self.args.K), 1])
    #         index = random.sample(range(0, self.args.K), m)
    #         self.dispatch(index)
    #         self.client_update(index)
    #         self.aggregation(index)
    #     print('Finished! Save all result file!')
    #     self.save_results_to_csv()
    #     return self.nn
    # def dispatch(self, index):
    #     for j in index:
    #         for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
    #             old_params.data = new_params.data.clone()
    def aggregation(self, index):
        s = 0
        for j in index:
            s += self.nns[j].len
        params = {}
        for k, v in self.nns[0].named_parameters():
            params[k] = torch.zeros_like(v.data)

        for j in index:
            for k, v in self.nns[j].named_parameters():
                params[k] += v.data * (self.nns[j].len / s)

        for k, v in self.nn.named_parameters():
            v.data = params[k].data.clone()


    def client_update(self, index):
        for k in index:
            self.nns[k], acc, emo_UF1, emo_UAR, train_loss, val_loss = train(self.args, self.nns[k], self.nn)
            self.acc_list[k].append(acc)
            self.uf1_list[k].append(emo_UF1)
            self.uar_list[k].append(emo_UAR)
            self.train_loss_list[k].append(train_loss)
            self.val_loss_list[k].append(val_loss)

    def generate_personalized_model(self, client_model, other_clients_models):

        aligned_models = []
        for other_client_model in other_clients_models:
            # aligned_model = ot_alignment([client_model, other_client_model], self.args.gpu_id)
            aligned_model = other_client_model
            aligned_models.append(aligned_model)

        personalized_model = copy.deepcopy(client_model)

        other_clients_data_sizes = [model.len for model in other_clients_models]

        total_other_clients_data_size = sum(other_clients_data_sizes)

        with torch.no_grad():
            client_params = {name: param for name, param in client_model.named_parameters()}

            aligned_params = [{name: param for name, param in aligned_model.named_parameters()}
                              for aligned_model in aligned_models]

            for name, param_personalized in personalized_model.named_parameters():
                param_personalized.data = client_params[name].data * 0.9

                for aligned_model_params, data_size in zip(aligned_params, other_clients_data_sizes):
                    weight = (data_size / total_other_clients_data_size) * 0.1
                    param_personalized.data += aligned_model_params[name].data * weight
        return personalized_model
    def save_results_to_csv(self):
        folder_name = f"fedprox/exp/exp_persornalize_fedprox/{self.args.save_path}/seed_{self.args.seed}"
        print(f'save path folder: {self.args.save_path}')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        model_casme2_data = {
            'acc': self.acc_list[0],
            'uf1': self.uf1_list[0],
            'uar': self.uar_list[0],
            'train_loss': self.train_loss_list[0],
            'val_loss': self.val_loss_list[0],
        }
        model_casme2_df = pd.DataFrame(model_casme2_data)
        model_casme2_df.to_csv(f'{folder_name}/model_casme2_result.csv', index=False)

        model_samm_data = {
            'acc': self.acc_list[1],
            'uf1': self.uf1_list[1],
            'uar': self.uar_list[1],
            'train_loss': self.train_loss_list[1],
            'val_loss': self.val_loss_list[1],
        }
        model_samm_df = pd.DataFrame(model_samm_data)
        model_samm_df.to_csv(f'{folder_name}/model_samm_result.csv', index=False)

        model_casme3_data = {
            'acc': self.acc_list[2],
            'uf1': self.uf1_list[2],
            'uar': self.uar_list[2],
            'train_loss': self.train_loss_list[2],
            'val_loss': self.val_loss_list[2],
        }
        model_casme3_df = pd.DataFrame(model_casme3_data)
        model_casme3_df.to_csv(f'{folder_name}/model_casme3_result.csv', index=False)
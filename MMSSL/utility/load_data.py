import json
import pickle
import random as rd
from time import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm
from utility.parser import parse_args

args = parse_args()


def load_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


class Data(object):
    def __init__(self, dataset: str = "amazon_clothing_8_core", batch_size: int = 128):
        dataset = dataset.split("_")
        dataset[-2] = dataset[-2] + "_" + dataset[-1]
        self.path = f"../../Grad_proj/sequential/data/{'/'.join(dataset[:-1])}"
        self.batch_size = batch_size

        metadata = load_json(f"{self.path}/uniqued_metadata.json")
        test = torch.load(f"{self.path}/uniqued_test_data.pt")
        valid = [[v[-2]] for v in test]
        train = [v[:-2] for v in test]
        test = [[v[-1]] for v in test]

        # get number of users and items
        self.n_users, self.n_items = metadata["num of user"], metadata["num of item"]
        self.n_train = sum([len(ele) for ele in train])
        self.n_valid = self.n_users
        self.n_test = self.n_users
        self.neg_pools = {}

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Item_Interacts = sp.dok_matrix(
            (self.n_items, self.n_items), dtype=np.float32
        )

        self.train_items, self.test_set, self.val_set = {}, {}, {}
        self.exist_users = list(range(self.n_users))

        row, col = [], []
        for uid, train_items in tqdm(enumerate(train), total=len(train)):
            uid = int(uid)
            for i in train_items:
                self.R[uid, i] = 1.0
                row.append(int(uid))
                col.append(int(i))

            self.train_items[uid] = train_items
        data = np.ones(len(row))
        train_mat = csr_matrix((data, (row, col)), shape=(self.n_users, self.n_items))
        pickle.dump(train_mat, open(f"{self.path}/train_mat", "wb"))

        row, col = [], []
        for uid, test_items in enumerate(test):
            uid = int(uid)
            if len(test_items) == 0:
                continue
            try:
                self.test_set[uid] = test_items
            except:
                continue

        for uid, val_items in enumerate(valid):
            uid = int(uid)
            if len(val_items) == 0:
                continue
            try:
                self.val_set[uid] = val_items
            except:
                continue

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + "/s_adj_mat.npz")
            norm_adj_mat = sp.load_npz(self.path + "/s_norm_adj_mat.npz")
            mean_adj_mat = sp.load_npz(self.path + "/s_mean_adj_mat.npz")
            print("already load adj matrix", adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + "/s_adj_mat.npz", adj_mat)
            sp.save_npz(self.path + "/s_norm_adj_mat.npz", norm_adj_mat)
            sp.save_npz(self.path + "/s_mean_adj_mat.npz", mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[: self.n_users, self.n_users :] = R
        adj_mat[self.n_users :, : self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print("already create adjacency matrix", adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print("generate single-normalized adjacency matrix.")
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print(
                "check normalized adjacency matrix whether equal to this laplacian matrix."
            )
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print("already normalize adjacency matrix", time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        return users, pos_items, neg_items

    def print_statistics(self):
        print("n_users=%d, n_items=%d" % (self.n_users, self.n_items))
        print("n_interactions=%d" % (self.n_train + self.n_test + self.n_valid))
        print(
            "n_train=%d, n_valid=%d ,n_test=%d, sparsity=%.5f"
            % (
                self.n_train,
                self.n_test,
                self.n_test,
                (self.n_train + self.n_test + self.n_valid)
                / (self.n_users * self.n_items),
            )
        )

    def json2mat(self):
        f = open("/home/weiw/Code/MM/MMSSL/data/clothing/train.json", "r")
        train = json.load(f)
        row, col = [], []
        for index, value in enumerate(train.keys()):
            for i in range(len(train[value])):
                row.append(int(value))
                col.append(train[value][i])
        data = np.ones(len(row))
        train_mat = csr_matrix((data, (row, col)), shape=(n_user, n_item))
        pickle.dump(train_mat, open("./train_mat", "wb"))

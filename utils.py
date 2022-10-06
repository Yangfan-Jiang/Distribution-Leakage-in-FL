"""
Implementation of several useful tool
Please copy this script to target path
"""
import copy
import torch
import random
import numpy as np
import pandas as pd
from torch import optim

from scipy.optimize import curve_fit
from scipy.stats import norm, rv_histogram, entropy, wasserstein_distance
from sklearn.metrics import mutual_info_score


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 30, 2000
    num_shards = int(num_users*3)
    num_imgs = int(60000 / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 3, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def synthetic_marginal(feature_dist, num_feature, max_samples=1000):
    """
    Generate samples according to marginal distribution
    feature_dist: dict ==> [(mean, std), ...]      ([value1, value2, ...], [p1, p2, ...]) for discrete data
    """
    marginal_dist_data = []
    for i in range(max_samples):
        # continues feature ~ N(0, 1)
        # we only need to modify discrete data 
        x = np.random.normal(0.5, 0.25, num_feature)
        for f in range(num_feature):
            dist_info = feature_dist[f]
            # discrete data
            if type(dist_info[0]) is np.ndarray:
                # x[f] = np.random.choice(dist_info[0], 1, p=dist_info[1])[0]
                x[f] = np.random.choice(dist_info[0], 1)[0]
                #if data_pth == 'uci_credit/' and f==3 and x[f] < -2:
                #    x[f] = 0.85855728
            else:
                pass
        marginal_dist_data.append(x)
    return marginal_dist_data


def synthetic_gaussian(feature_dist, num_feature, max_samples=1000):
    """Baseline Method"""
    marginal_dist_data = []
    for i in range(max_samples):
        # continues feature ~ Uni(-2, 2)
        # we only need to modify discrete data 
        x = np.random.normal(0.5, 0.25, num_feature)
        for f in range(num_feature):
            dist_info = feature_dist[f]
            # discrete data
            if type(dist_info[0]) is np.ndarray:
                x[f] = np.random.choice(dist_info[0], 1)[0]
        marginal_dist_data.append(x)
    return marginal_dist_data


def synthetic_uniform(feature_dist, num_feature, max_samples=1000):
    """Baseline Method"""
    marginal_dist_data = []
    for i in range(max_samples):
        # continues feature ~ Uni(-2, 2)
        # we only need to modify discrete data 
        x = np.random.uniform(0, 1, num_feature)
        for f in range(num_feature):
            dist_info = feature_dist[f]
            # discrete data
            if type(dist_info[0]) is np.ndarray:
                x[f] = np.random.choice(dist_info[0], 1)[0]
                #if data_pth == 'uci_credit/' and f==3 and x[f] < -2:
                #    x[f] = 0.85855728
            else:
                pass
        marginal_dist_data.append(x)
    return marginal_dist_data
    

def compute_loss_from_real(sample, ml):
    data_dist = [[] for _ in range(len(ml))]
    loss_fn = nn.CrossEntropyLoss()
    for ids,s in enumerate(sample):
        loss_m = []
        for i, m in enumerate(ml):
            fake_y = torch.max(g(torch.tensor(s).float()), 0)[1].unsqueeze(0)
            pred = m(torch.tensor(s).float()).unsqueeze(0)
            tmp_loss = loss_fn(pred, fake_y.clone().detach().long()).item()
            loss_m.append(tmp_loss)
        loss_m = np.array(loss_m)
        min_id = np.argmin(loss_m)
        data_dist[min_id].append(s)
    return data_dist


def compute_confidence_score(sample, ml, g):
    data_dist = [[] for _ in range(len(ml))]
    # loss_fn = nn.CrossEntropyLoss()
    for ids, s in enumerate(sample):
        dist_m = []
        tmps = torch.tensor(s).float()
        v_g = torch.softmax(g(tmps),0).detach().numpy()
        for i, m in enumerate(ml):
            v = torch.softmax(m(tmps),0).detach().numpy()
            w_dist = wasserstein_distance(v, v_g)
            dist_m.append(w_dist)
        dist_m = np.array(dist_m)
        #print(dist_m)
        min_id = np.argmin(dist_m)
        data_dist[min_id].append(s)
    return data_dist


def synthetic_copula(size_ratio, num_feature, global_x, discrete_feature, dist_info, fake_sample_size=5000):
    aux_sample = []
    sample_size = int(global_x.shape[0] * 5 * size_ratio)
    idx = np.random.choice(list(range(global_x.shape[0])), sample_size, replace=False)
    curr_sample = copy.deepcopy(global_x[idx])
    for i, s in enumerate(curr_sample):
        aux_sample.append(curr_sample[i])
    aux_sample = np.array(aux_sample)

    df_sample = pd.DataFrame(aux_sample, columns=[str(i) for i in range(num_feature)])
    sdv_model = GaussianCopula()
    sdv_model.fit(df_sample)
    new_data = np.array(sdv_model.sample(fake_sample_size - sample_size))
    
    # discrete featur
    for fid in discrete_feature:
        for s in new_data:
            s[fid] = dist_info[fid][0][np.argmin(abs(dist_info[fid][0] - s[fid]))]
    copula_sample = np.concatenate((aux_sample, new_data), axis=0)
    return copula_sample, aux_sample


def plot_coor_heatmap():
    df_global = pd.DataFrame(global_x, columns=[str(i) for i in range(num_feature)])
    df_sample = pd.DataFrame(sample, columns=[str(i) for i in range(num_feature)])

    df_fake_sample = pd.DataFrame(fake_sample, columns=[str(i) for i in range(num_feature)])
    df_aug_sample = pd.DataFrame(new_data, columns=[str(i) for i in range(num_feature)])
    sns.heatmap(df_global.corr(),cmap="YlGnBu",xticklabels=2)
    plt.tight_layout()
    plt.show()
    sns.heatmap(df_sample.corr(),cmap="YlGnBu",xticklabels=2)
    plt.tight_layout()
    plt.show()
    sns.heatmap(df_aug_sample.corr(),cmap="YlGnBu",xticklabels=2)
    plt.tight_layout()
    plt.show()
    sns.heatmap(df_fake_sample.corr(),cmap="YlGnBu",xticklabels=2)
    plt.tight_layout()
    plt.show()
    
    
def synthesis_model_diff(c, feature_dist, num_feature, ml, g, prior_dist, threshold):
    """
    c: client
    kmax = 128
    rejmax = 10
    kmin = 4
    confmin = 0.2
    """
    # continues feature ~ N(0, 1)
    # we only need to modify discrete data 
    def random_value(prior_dist):
        if prior_dist == 'uniform':
            return np.random.uniform(r_min, r_max)
        elif prior_dist == 'gaussian':
            return np.random.normal(0.5, 0.25)
            
    r_min = 0
    r_max = 1
    x = np.random.uniform(r_min, r_max, num_feature)
    for f in range(num_feature):
        x[f] = random_value(prior_dist)
        # discrete data
        if type(feature_dist[f][0]) is np.ndarray:
            x[f] = np.random.choice(feature_dist[f][0], 1)[0]
        
    y_star = 0
    diff_star = 999
    j = 0
    k = int(num_feature/2)   # 128
    rejmax = 10
    kmin = 4
    # threshold = threshold
    
    for _ in range(1000):
        tmpx = torch.tensor(x).float()
        yg = torch.softmax(g(tmpx),0) #.detach().numpy()
        yc = torch.softmax(ml[c](tmpx),0) #.detach().numpy()
        diff = torch.nn.MSELoss()(yg, yc) + 0.2*tmpx.var()
        # print(wasserstein_distance(yg, yc), tmpx.var())

        if diff <= diff_star:
            if diff <= threshold:
                #print("iter=", _)
                return x
            x_star = x
            diff_star = diff
            j = 0
        j += 1
        if j > rejmax:
            k = int(max(kmin, np.ceil(k/2)))
            j = 0
        # x <-- randrecord(x*, k)
        # select k features randomly
        feature_id = np.random.choice(num_feature, k, replace=False)
        x_star_tmp = copy.deepcopy(x_star)
        for f in feature_id:
            x_star_tmp[f] = random_value(prior_dist)
            # discrete data
            if type(feature_dist[f][0]) is np.ndarray:
                x_star_tmp[f] = np.random.choice(feature_dist[f][0], 1)[0]
        x = x_star_tmp
    return None


def hc_attack_gbdt(c, feature_dist, num_feature, ml, prior_dist, threshold):
    """
    c: client
    kmax = 128
    rejmax = 10
    kmin = 4
    confmin = 0.2
    """
    #def g(x):
    #    pred_prob = np.mean([local_ml.predict_proba(x.reshape(1, -1)) for local_ml in ml], 0)[0]
    #    return pred_prob
    
    # continues feature ~ N(0, 1)
    # we only need to modify discrete data 
    def random_value(prior_dist):
        if prior_dist == 'uniform':
            return np.random.uniform(r_min, r_max)
        elif prior_dist == 'gaussian':
            return np.random.normal(0.5, 0.25)
            
    r_min = 0
    r_max = 1
    x = np.random.uniform(r_min, r_max, num_feature)
    for f in range(num_feature):
        x[f] = random_value(prior_dist)
        # discrete data
        if type(feature_dist[f][0]) is np.ndarray:
            x[f] = np.random.choice(feature_dist[f][0], 1)[0]
        
    y_star = 0
    diff_star = 999
    j = 0
    k = int(num_feature/2)   # 128
    rejmax = 10
    kmin = 4
    # threshold = threshold
    
    for _ in range(1000):
        yg = np.mean([local_ml.predict_proba(x.reshape(1, -1)) for local_ml in ml], 0)[0]
        yc = ml[c].predict_proba(x.reshape(1, -1))[0]
        # print(yg,yc)
        diff = torch.nn.MSELoss()(torch.tensor(yg), torch.tensor(yc)) + 0.2*x.var()
        # print(wasserstein_distance(yg, yc), x.var())

        if diff <= diff_star:
            if diff <= threshold:
                #print("iter=", _)
                return x
            x_star = x
            diff_star = diff
            j = 0
        j += 1
        if j > rejmax:
            k = int(max(kmin, np.ceil(k/2)))
            j = 0
        # x <-- randrecord(x*, k)
        # select k features randomly
        feature_id = np.random.choice(num_feature, k, replace=False)
        x_star_tmp = copy.deepcopy(x_star)
        for f in feature_id:
            x_star_tmp[f] = random_value(prior_dist)
            # discrete data
            if type(feature_dist[f][0]) is np.ndarray:
                x_star_tmp[f] = np.random.choice(feature_dist[f][0], 1)[0]
        x = x_star_tmp
    return None


def hc_attack(c, feature_dist, num_feature, ml, g, prior_dist, threshold):
    """
    c: client
    kmax = 128
    rejmax = 10
    kmin = 4
    confmin = 0.2
    """
    #def g(x):
    #    pred_prob = np.mean([local_ml.predict_proba(x.reshape(1, -1)) for local_ml in ml], 0)[0]
    #    return pred_prob
    
    # continues feature ~ N(0, 1)
    # we only need to modify discrete data 
    def random_value(prior_dist):
        if prior_dist == 'uniform':
            return np.random.uniform(r_min, r_max)
        elif prior_dist == 'gaussian':
            return np.random.normal(0.5, 0.25)
            
    r_min = 0
    r_max = 1
    x = np.random.uniform(r_min, r_max, num_feature)
    for f in range(num_feature):
        x[f] = random_value(prior_dist)
        # discrete data
        if type(feature_dist[f][0]) is np.ndarray:
            x[f] = np.random.choice(feature_dist[f][0], 1)[0]
        
    y_star = 0
    diff_star = 999
    j = 0
    k = int(num_feature/2)   # 128
    rejmax = 10
    kmin = 4
    # threshold = threshold
    
    for _ in range(1000):
        tmpx = torch.tensor(x).float()
        yg = torch.softmax(g(tmpx),0) #.detach().numpy()
        yc = torch.softmax(ml[c](tmpx),0) #.detach().numpy()
        diff = torch.nn.MSELoss()(yg, yc) + 0.1*tmpx.var()

        if diff <= diff_star:
            if diff <= threshold:
                #print("iter=", _)
                return x
            x_star = x
            diff_star = diff
            j = 0
        j += 1
        if j > rejmax:
            k = int(max(kmin, np.ceil(k/2)))
            j = 0
        # x <-- randrecord(x*, k)
        # select k features randomly
        feature_id = np.random.choice(num_feature, k, replace=False)
        x_star_tmp = copy.deepcopy(x_star)
        for f in feature_id:
            x_star_tmp[f] = random_value(prior_dist)
            # discrete data
            if type(feature_dist[f][0]) is np.ndarray:
                x_star_tmp[f] = np.random.choice(feature_dist[f][0], 1)[0]
        x = x_star_tmp
    return None


def synthesis_by_model(c, feature_dist):
    """
    c: class
    kmax = 128
    rejmax = 10
    kmin = 4
    confmin = 0.2
    """
    # continues feature ~ N(0, 1)
    # we only need to modify discrete data 
    #x = np.random.uniform(-2, 2, num_feature)
    x = np.random.normal(0, 1, num_feature)
    for f in range(num_feature):
        dist_info = feature_dist[f]
        # discrete data
        if type(dist_info[0]) is np.ndarray:
            
            x[f] = np.random.choice(dist_info[0], 1)[0]
            
    y_star = 0
    j = 0
    k = num_feature   # 128
    rejmax = 10
    kmin = 4
    confmin = 0.5
    
    for _ in range(1000):
        tmpx = torch.tensor(x).float()
        y = torch.softmax(g(tmpx),0).detach().numpy()
        #print(y[c])
        if y[c] >= y_star:
            if y[c] > confmin and c == np.argmax(y):
                if np.random.random() < y[c]:
                    #print("iter=", _)
                    return x
            x_star = x
            y_star = y[c]
            j = 0
        j += 1
        if j > rejmax:
            k = int(max(kmin, np.ceil(k/2)))
            j = 0
        # x <-- randrecord(x*, k)
        # select k features randomly
        feature_id = np.random.choice(num_feature, k, replace=False)
        x_star_tmp = copy.deepcopy(x_star)
        for f in feature_id:
            x_star_tmp[f] = np.random.normal(0, 1)
            dist_info = feature_dist[f]
            # discrete data
            if type(dist_info[0]) is np.ndarray:
                x_star_tmp[f] = np.random.choice(dist_info[0], 1, p=dist_info[1])[0]
                # x_star[f] = np.random.choice(dist_info[0], 1)[0]
        x = x_star_tmp
    print("none", c)
    return None


def gen_synthesis_by_model(feature_dist, fake_sample_size):
    fake_sample_size = int(fake_sample_size / num_class)
    sample_list = []
    for c in range(num_class):
        for i in range(fake_sample_size):
            x = synthesis_by_model(c, dist_info)
            while x is None:
                x = synthesis_by_model(c, dist_info)
            sample_list.append(x)
    return sample_list


def gen_synthesis_model_diff(dist_info, num_feature, num_client, fake_sample_size, ml, g, prior_dist):
    fake_sample_size = int(fake_sample_size / num_client)
    sample_list = []
    for c in range(num_client):
        tmpsample = []
        threshold = 1e-3
        for i in range(fake_sample_size):
            x = synthesis_model_diff(c, dist_info, num_feature, ml, g, prior_dist, threshold)
            j = 0
            while x is None:
                x = synthesis_model_diff(c, dist_info, num_feature, ml, g, prior_dist, threshold)
                j += 1
                if j > 10:
                    threshold *= 1.5
                    print("threshold =", threshold)
                    j = 0
            tmpsample.append(copy.deepcopy(x))
            if len(tmpsample)%400 == 0:
                print(len(tmpsample))
        sample_list.append(copy.deepcopy(np.array(tmpsample)))
    return sample_list
    


def synthetic_marginal_f(data, att_feature, per_samples=100):
    """
    Generate samples according to marginal distribution
    feature_dist: dict ==> [(mean, std), ...]      ([value1, value2, ...], [p1, p2, ...]) for discrete data
    """
    marginal_dist_data = []
    for d in data:
        for _ in range(per_samples):
            d[att_feature] = np.random.normal(0, 1)
            marginal_dist_data.append(copy.deepcopy(d))
    return marginal_dist_data
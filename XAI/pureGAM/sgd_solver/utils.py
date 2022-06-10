import numpy as np
import torch as th


def save_model(enc_cate, enc_num, model_cate, model_smooth, save_path, isInfo=True):
    state = {
        #"enc_cate": enc_cate.state_dict(),
        "enc_num": enc_num.state_dict(),
        "model_cate": model_cate.state_dict(),
        "model_smooth": model_smooth.state_dict(),
    }
    if isInfo:
        print("[Info] Save_model to .....  ", save_path)
    th.save(state, save_path)


def load_model(enc_cate, enc_num, model_cate, model_smooth, load_path):
    print("[Info] Load model from ....   ", load_path)
    checkpoint = th.load(load_path)
    #enc_cate.load_state_dict(checkpoint['enc_cate'])
    enc_num.load_state_dict(checkpoint['enc_num'])
    model_cate.load_state_dict(checkpoint['model_cate'])
    model_smooth.load_state_dict(checkpoint['model_smooth'])
    #self.n_epoch = checkpoint['epoch']
    return enc_cate, enc_num, model_cate, model_smooth

def generate_pairwise_idxes(n):
    assert n >= 0
    ret_lst = []
    if n < 2:
        return ret_lst
    for i in range(n):
        for j in range(i+1, n):
            ret_lst.append([i, j])
    return ret_lst


def _print_metrics(lst):
    return np.around(lst, 5)

# deal with a line with all zeros, in case t_sum == 0

# in case there are point far away, should consider t_sum==0

'''if (t_sum == 0).any():
    print("BOOM!")
    print(t_sum[-3],t_sum[-3].max(), t_sum[-3]. min())
    print(t_sum[-4],t_sum[-4].max(), t_sum[-4]. min())
    print(t_sum[-2],t_sum[-2].max(), t_sum[-2]. min())
    print(t[-3],t[-3].max(), t[-3]. min())
    print(t[-4],t[-4].max(), t[-4]. min())
    print(t[-2],t[-2].max(), t[-2]. min())
    assert False'''
eps = 1e-13
def safe_norm(t: th.tensor)->th.tensor:
    t_sum = t.sum(axis=-1, keepdim=True)
    t /= (t_sum + eps)
    return th.where(th.isnan(t), th.full_like(t, 0), t)

def safe_norm_alter(t: th.tensor)->th.tensor:
    # norm using sum of all matrix rather than sum of an axis.
    t_sum = t.sum(axis=-1, keepdim=True).sum(axis=-2, keepdim=True)
    # multi  t.shape[-1] to keep in the same scale
    t *= t.shape[-1] / (t_sum + eps)
    return th.where(th.isnan(t), th.full_like(t, 0), t)



'''eps = 1e-15
def safe_norm(t: th.tensor)->th.tensor:
    t += eps
    t_sum = t.sum(axis=-1, keepdim=True)
    t /= t_sum
    return th.where(th.isnan(t), th.full_like(t, 0), t)

def safe_norm_alter(t: th.tensor)->th.tensor:
    t += eps
    # norm using sum of all matrix rather than sum of an axis.
    t_sum = t.sum(axis=-1, keepdim=True).sum(axis=-2, keepdim=True)
    # multi  t.shape[-1] to keep in the same scale
    t *= t.shape[-1] / t_sum
    return th.where(th.isnan(t), th.full_like(t, 0), t)'''
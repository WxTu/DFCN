import opt
import torch
import numpy as np
from DFCN import DFCN
from utils import setup_seed
from sklearn.decomposition import PCA
from load_data import LoadDataset, load_graph, construct_graph
from train import Train, acc_reuslt, nmi_result, f1_result, ari_result

setup_seed(opt.args.seed)

print("network setting…")

if opt.args.name == 'usps':
    opt.args.k = 5
    opt.args.n_clusters = 10
    opt.args.n_input = 30
elif opt.args.name == 'hhar':
    opt.args.k = 5
    opt.args.n_clusters = 6
    opt.args.n_input = 50
elif opt.args.name == 'reut':
    opt.args.k = 5
    opt.args.n_clusters = 4
    opt.args.n_input = 100
elif opt.args.name == 'acm':
    opt.args.k = None
    opt.args.n_clusters = 3
    opt.args.n_input = 100
elif opt.args.name == 'dblp':
    opt.args.k = None
    opt.args.n_clusters = 4
    opt.args.n_input = 50
elif opt.args.name == 'cite':
    opt.args.k = None
    opt.args.n_clusters = 6
    opt.args.n_input = 100
else:
    print("error!")

### cuda
print("use cuda: {}".format(opt.args.cuda))
device = torch.device("cuda" if opt.args.cuda else "cpu")

### root
opt.args.data_path = 'data/{}.txt'.format(opt.args.name)
opt.args.label_path = 'data/{}_label.txt'.format(opt.args.name)
opt.args.graph_k_save_path = 'graph/{}{}_graph.txt'.format(opt.args.name, opt.args.k)
opt.args.graph_save_path = 'graph/{}_graph.txt'.format(opt.args.name)
opt.args.pre_model_save_path = 'model/model_pretrain/{}_pretrain.pkl'.format(opt.args.name)
opt.args.final_model_save_path = 'model/model_final/{}_final.pkl'.format(opt.args.name)

### data pre-processing
print("Data: {}".format(opt.args.data_path))
print("Label: {}".format(opt.args.label_path))

graph = ['acm', 'dblp', 'cite']
non_graph = ['usps', 'hhar', 'reut']

x = np.loadtxt(opt.args.data_path, dtype=float)
y = np.loadtxt(opt.args.label_path, dtype=int)

pca = PCA(n_components=opt.args.n_input)
X_pca = pca.fit_transform(x)
# plot_pca_scatter(args.name, args.n_clusters, X_pca, y)

dataset = LoadDataset(X_pca)

if opt.args.name in non_graph:
    construct_graph(opt.args.graph_k_save_path, X_pca, y, 'heat', topk=opt.args.k)

adj = load_graph(opt.args.k, opt.args.graph_k_save_path, opt.args.graph_save_path, opt.args.data_path).to(device)
data = torch.Tensor(dataset.x).to(device)
label = y

###  model definition
model = DFCN(ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2, ae_n_enc_3=opt.args.ae_n_enc_3,
             ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2, ae_n_dec_3=opt.args.ae_n_dec_3,
             gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2, gae_n_enc_3=opt.args.gae_n_enc_3,
             gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2, gae_n_dec_3=opt.args.gae_n_dec_3,
             n_input=opt.args.n_input,
             n_z=opt.args.n_z,
             n_clusters=opt.args.n_clusters,
             v=opt.args.freedom_degree,
             n_node=data.size()[0],
             device=device).to(device)

### training
print("Training on {}…".format(opt.args.name))
if opt.args.name == "usps":
    lr = opt.args.lr_usps
elif opt.args.name == "hhar":
    lr = opt.args.lr_hhar
elif opt.args.name == "reut":
    lr = opt.args.lr_reut
elif opt.args.name == "acm":
    lr = opt.args.lr_acm
elif opt.args.name == "dblp":
    lr = opt.args.lr_dblp
elif opt.args.name == "cite":
    lr = opt.args.lr_cite
else:
    print("missing lr!")

Train(opt.args.epoch, model, data, adj, label, lr, opt.args.pre_model_save_path, opt.args.final_model_save_path,
      opt.args.n_clusters, opt.args.acc, opt.args.gamma_value, opt.args.lambda_value, device)


print("ACC: {:.4f}".format(max(acc_reuslt)))
print("NMI: {:.4f}".format(nmi_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
print("ARI: {:.4f}".format(ari_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
print("F1: {:.4f}".format(f1_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
print("Epoch:", np.where(acc_reuslt == np.max(acc_reuslt))[0][0])

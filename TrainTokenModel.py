import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup, AdamW
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import logging
from model import TwinBert, SiameseNetworkDataset, CosineContrastiveLoss, BingDataset, CosineCrossEntropyLoss, BingGraphDataset, GraphTwinBert, TokenModel
from tensorboardX import SummaryWriter
import time
from utils import load_data, create_graph, load_data_new, load_data_large, set_seed
from parse import parse_args
import dgl
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

args = parse_args()
print(args)
set_seed(args.seed)
logging.basicConfig(level=logging.ERROR)
MAX_LEN = args.max_len
TRAIN_BATCH_SIZE = args.train_batch
VALID_BATCH_SIZE = args.val_batch
EPOCHS = args.epoch
LEARNING_RATE = args.lr
writer = SummaryWriter('runs/' + time.strftime("%Y%m%d%H%M%S", time.localtime()))
global global_iter
global_iter = 0
encoder = args.encoder

data_args = {'type': args.dataset_type, 'topk': args.topk, 'threshold': args.threshold, 'randk': args.randk}
# train_dataset, test_dataset = load_data_new(dataset=args.dataset, **data_args)
train_dataset, val_dataset, test_dataset = load_data_large(dataset=args.dataset, **data_args)
train_dataset = create_graph(train_dataset)
val_dataset = create_graph(val_dataset)
test_dataset = create_graph(test_dataset)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
training_set = BingGraphDataset(train_dataset, tokenizer, MAX_LEN)
val_set = BingGraphDataset(val_dataset, tokenizer, MAX_LEN)
testing_set = BingGraphDataset(test_dataset, tokenizer, MAX_LEN)

gpus = [0, 1, 2, 3]
device = gpus[0]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))
model = TokenModel(dropout=args.dropout, encoder=encoder, with_description=args.with_description, use_domain_emb=args.use_domain_emb)
model.to(device)
# model = nn.DataParallel(model, device_ids=gpus, output_device=device)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

val_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

# training_loader = DataLoader(training_set, **train_params)
# testing_loader = DataLoader(testing_set, **test_params)

training_loader = GraphDataLoader(training_set, **train_params)
val_loader = GraphDataLoader(val_set, **test_params)
testing_loader = GraphDataLoader(testing_set, **test_params)

# criterion = CosineContrastiveLoss()
# criterion = CosineCrossEntropyLoss()
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay' : args.weight_decay
    },
    {'params' : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay' : 0.0
    }
]
# optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE, weight_decay=args.weight_decay)
optimizer = AdamW(optimizer_grouped_parameters,lr = LEARNING_RATE)
print(model)

epochs = 2
total_steps = len(training_loader) * epochs


def train(epoch):
    global global_iter
    model.train()
    fin_targets=[]
    fin_outputs=[]
    loss_mean = 0.
    for data in training_loader:
    # for data in tqdm(training_loader):
        graphs, labels, q_ids_neighbor, q_mask_neighbor, q_token_type_neighbor, d_ids_neighbor, d_mask_neighbor, d_token_type_neighbor, domain_idx = data
        # ids, mask, token_type_ids, graphs = data['ids'], data['mask'], data['token_type_ids'], data['graphs'] 
        domain_idx = domain_idx.to(device, dtype = torch.long)
        graphs = graphs.to(device)
        targets = labels.to(device, dtype = torch.long)
        q_ids_neighbor = q_ids_neighbor.to(device)
        q_mask_neighbor = q_mask_neighbor.to(device)
        q_token_type_neighbor = q_token_type_neighbor.to(device)

        d_ids_neighbor = d_ids_neighbor.to(device)
        d_mask_neighbor = d_mask_neighbor.to(device)
        d_token_type_neighbor = d_token_type_neighbor.to(device)

        # graph = dgl.batch(graphs).to(device)
        logits = model(graphs, q_ids_neighbor, q_mask_neighbor, q_token_type_neighbor, d_ids_neighbor, d_mask_neighbor, d_token_type_neighbor, domain_idx)
        # prediction = model.module.predict(output1, output2)
        prediction = F.softmax(logits, dim=1)
        loss = criterion(logits.view(-1, 2), targets.view(-1))
        # loss = criterion(output1,output2,targets)
        writer.add_scalar('loss', loss, global_step=global_iter)
        global_iter += 1

        # batch eval
        fin_targets.extend(targets.cpu().detach().numpy().tolist())
        fin_outputs.extend(prediction[:, 1].cpu().detach().numpy().tolist())
        
        loss_mean += loss.item() * len(targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print(sum(np.array(fin_outputs) >= 0.5))
    # print(sum(fin_targets))
    fin_outputs = np.array(fin_outputs)
    y_pred = np.array(fin_outputs) >= 0.5
    acc = metrics.accuracy_score(fin_targets, y_pred)
    f1_score_macro = metrics.f1_score(fin_targets,  y_pred, average='macro')
    auc = metrics.roc_auc_score(fin_targets, fin_outputs)

    loss_mean = loss_mean / len(fin_targets)
    print(f'Epoch: {epoch}, Loss:  {loss_mean}')
    return acc, f1_score_macro, auc

def validation():
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for data in val_loader:
        # for data in tqdm(val_loader):
            graphs, labels, q_ids_neighbor, q_mask_neighbor, q_token_type_neighbor, d_ids_neighbor, d_mask_neighbor, d_token_type_neighbor, domain_idx = data
            domain_idx = domain_idx.to(device, dtype = torch.long)
            graphs = graphs.to(device)
            targets = labels.to(device, dtype = torch.long)
            q_ids_neighbor = q_ids_neighbor.to(device)
            q_mask_neighbor = q_mask_neighbor.to(device)
            q_token_type_neighbor = q_token_type_neighbor.to(device)

            d_ids_neighbor = d_ids_neighbor.to(device)
            d_mask_neighbor = d_mask_neighbor.to(device)
            d_token_type_neighbor = d_token_type_neighbor.to(device)

            logits = model(graphs, q_ids_neighbor, q_mask_neighbor, q_token_type_neighbor, d_ids_neighbor, d_mask_neighbor, d_token_type_neighbor, domain_idx)
            prediction = F.softmax(logits, dim=1)
            # prediction = model.module.predict(output1, output2)
            # cos_sim = F.cosine_similarity(output1, output2)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(prediction[:, 1].cpu().detach().numpy().tolist())
    fin_outputs = np.array(fin_outputs)
    y_pred = np.array(fin_outputs) >= 0.5
    acc = metrics.accuracy_score(fin_targets, y_pred)
    f1_score_macro = metrics.f1_score(fin_targets,  y_pred, average='macro')
    auc = metrics.roc_auc_score(fin_targets, fin_outputs)
    return acc, f1_score_macro, auc

def test():
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for data in testing_loader:
        # for data in tqdm(testing_loader):
            graphs, labels, q_ids_neighbor, q_mask_neighbor, q_token_type_neighbor, d_ids_neighbor, d_mask_neighbor, d_token_type_neighbor, domain_idx = data
            domain_idx = domain_idx.to(device, dtype = torch.long)
            graphs = graphs.to(device)
            targets = labels.to(device, dtype = torch.long)
            q_ids_neighbor = q_ids_neighbor.to(device)
            q_mask_neighbor = q_mask_neighbor.to(device)
            q_token_type_neighbor = q_token_type_neighbor.to(device)

            d_ids_neighbor = d_ids_neighbor.to(device)
            d_mask_neighbor = d_mask_neighbor.to(device)
            d_token_type_neighbor = d_token_type_neighbor.to(device)

            logits = model(graphs, q_ids_neighbor, q_mask_neighbor, q_token_type_neighbor, d_ids_neighbor, d_mask_neighbor, d_token_type_neighbor, domain_idx)
            prediction = F.softmax(logits, dim=1)
            # prediction = model.module.predict(output1, output2)
            # cos_sim = F.cosine_similarity(output1, output2)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(prediction[:, 1].cpu().detach().numpy().tolist())
    fin_outputs = np.array(fin_outputs)
    y_pred = np.array(fin_outputs) >= 0.5
    acc = metrics.accuracy_score(fin_targets, y_pred)
    f1_score_macro = metrics.f1_score(fin_targets,  y_pred, average='macro')
    auc = metrics.roc_auc_score(fin_targets, fin_outputs)
    return acc, f1_score_macro, auc, y_pred.tolist()


best_val_auc = 0
best_test_auc, best_test_f1, best_acc = 0, 0, 0
count = 0
best_y_pred = []
for epoch in range(EPOCHS):
    print(f"=> Train epoch {epoch}")
    accuracy_train, f1_score_macro_train, auc_train = train(epoch)
    writer.add_scalar('train/accuracy', accuracy_train, global_step=(epoch))
    writer.add_scalar('train/f1 macro', f1_score_macro_train, global_step=(epoch))
    writer.add_scalar('train/auc', auc_train, global_step=(epoch))
    print(f"(train): Accuracy Score = {accuracy_train}, F1 Score (Macro) = {f1_score_macro_train}, Auc Score = {auc_train}")

    print(f"=> Validation epoch {epoch}")
    accuracy_val, f1_score_macro_val, auc_val = validation()
    writer.add_scalar('val/accuracy', accuracy_val, global_step=(epoch))
    writer.add_scalar('val/f1 macro', f1_score_macro_val, global_step=(epoch))
    writer.add_scalar('val/auc', auc_val, global_step=(epoch))
    print(f"(val): Accuracy Score = {accuracy_val}, F1 Score (Macro) = {f1_score_macro_val}, Auc Score = {auc_val}")

    print(f"=> Test epoch {epoch}")
    accuracy_test, f1_score_macro_test, auc_test, y_pred = test()
    writer.add_scalar('test/accuracy', accuracy_test, global_step=(epoch))
    writer.add_scalar('test/f1 macro', f1_score_macro_test, global_step=(epoch))
    writer.add_scalar('test/auc', auc_test, global_step=(epoch))
    print(f"(test): Accuracy Score = {accuracy_test}, F1 Score (Macro) = {f1_score_macro_test}, Auc Score = {auc_test}")

    if auc_val > best_val_auc:
        best_val_auc = auc_val
        best_test_auc, best_test_f1, best_acc = auc_test, f1_score_macro_test, accuracy_test
        best_y_pred = y_pred
        count = 0
    else:
        count += 1
    if count >= 3:
        print("Optimization Finished!")
        print(f"Best accuracy: {best_acc}, Best F1 Score (Macro): {best_test_f1}, Best Auc Score: {best_test_auc}")
        # test_data = pd.read_csv("data/test.tsv", sep='\t') 
        # test_data['3_neighbors'] = best_y_pred
        # test_data.to_csv("test.tsv", sep='\t')

# np.savetxt("label.txt", np.expand_dims(y_test_pred, 1))
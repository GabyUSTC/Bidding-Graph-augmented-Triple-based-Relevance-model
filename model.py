import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import transformers
from transformers import BertTokenizer, BertModel, BertConfig, DistilBertConfig, DistilBertModel, DistilBertTokenizer, RobertaConfig
from dgl.data import DGLDataset
from dgl.nn import GATConv, GraphConv, SAGEConv
import dgl
from token_model import GraphBertModel
import numpy as np

class SiameseNetworkDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.question1 = dataframe.question1
        self.question2 = dataframe.question2
        self.targets = dataframe.is_duplicate
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    
    def tokenize(self,input_text):
        input_text = " ".join(input_text.split())

        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return ids,mask,token_type_ids

    def __getitem__(self, index):
        ids1,mask1,token_type_ids1 = self.tokenize(str(self.question1[index]))
        ids2,mask2,token_type_ids2 = self.tokenize(str(self.question2[index]))
        


        return {
            'ids': [torch.tensor(ids1, dtype=torch.long),torch.tensor(ids2, dtype=torch.long)],
            'mask': [torch.tensor(mask1, dtype=torch.long),torch.tensor(mask2, dtype=torch.long)],
            'token_type_ids': [torch.tensor(token_type_ids1, dtype=torch.long),torch.tensor(token_type_ids2, dtype=torch.long)],
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class BingDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.query = dataframe['query']
        self.keyword = dataframe.keyword
        self.targets = dataframe.label
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    
    def tokenize(self,input_text):
        input_text = " ".join(input_text.split())

        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return ids, mask, token_type_ids

    def __getitem__(self, index):
        ids1, mask1, token_type_ids1 = self.tokenize(str(self.query[index]))
        ids2, mask2, token_type_ids2 = self.tokenize(str(self.keyword[index]))

        return {
            'ids': [torch.tensor(ids1, dtype=torch.long),torch.tensor(ids2, dtype=torch.long)],
            'mask': [torch.tensor(mask1, dtype=torch.long),torch.tensor(mask2, dtype=torch.long)],
            'token_type_ids': [torch.tensor(token_type_ids1, dtype=torch.long),torch.tensor(token_type_ids2, dtype=torch.long)],
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class BingGraphDataset(DGLDataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.query = dataframe['query']
        self.keyword = dataframe['keyword']
        self.neighbors = dataframe['neighbors']
        self.graph = dataframe['graph']
        self.targets = dataframe['label']
        self.description = dataframe['description']
        self.domain_idx = dataframe['domain_idx']
        self.max_len = max_len
        super(BingGraphDataset, self).__init__(name='bingdataset')
        # self.graphs, self.label = self._load_graph()

    def process(self):
        self.graphs, self.label = self._load_graph()
        
    def _load_graph(self):
        labels = torch.tensor(self.targets, dtype=torch.float)
        graphs = []
        for i in range(len(self.graph)):
            g = self.graph[i]
            g.query = self.query[i]
            g.keyword = [self.keyword[i]]
            g.keyword.extend(self.neighbors[i])
            graphs.append(g)
        return graphs, labels

    def __len__(self):
        return len(self.data['query'])
    
    
    def tokenize(self,input_text):
        input_text = " ".join(input_text.split())

        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return ids,mask,token_type_ids

    '''
    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(str(graph_path), self.graphs, {'labels': self.label})
    '''

    
    def __getitem__(self, idx):
        r""" Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (:class:`dgl.DGLGraph`, Tensor, Tensor, Tensor, Tensor)
        """
        ids1, mask1, token_type_ids1 = self.tokenize(str(self.query[idx]))

        ids_d, mask_d, token_type_ids_d = self.tokenize(str(self.description[idx]))

        ids_neighbor, mask_neighbor, token_type_neighbor, mask_readout = [], [], [], []

        for i, n in enumerate(self.neighbors[idx]):
            ids, mask, token_type_ids = self.tokenize(str(n))
            ids_neighbor.append(torch.tensor(ids, dtype=torch.long))
            mask_neighbor.append(torch.tensor(mask, dtype=torch.long))
            token_type_neighbor.append(torch.tensor(token_type_ids, dtype=torch.long))

            if i == 0:
                mask_readout.append(torch.tensor(1, dtype=torch.long))
            else:
                mask_readout.append(torch.tensor(0, dtype=torch.long))
        
        g = self.graphs[idx]
        g.ndata['ids_neighbor'] = torch.stack(ids_neighbor)
        g.ndata['mask_neighbor'] = torch.stack(mask_neighbor)
        g.ndata['token_type_neighbor'] = torch.stack(token_type_neighbor)
        g.ndata['mask_readout'] = torch.stack(mask_readout).unsqueeze(-1)

        return g, self.label[idx], torch.tensor(ids1, dtype=torch.long), torch.tensor(mask1, dtype=torch.long), torch.tensor(token_type_ids1, dtype=torch.long),\
            torch.tensor(ids_d, dtype=torch.long), torch.tensor(mask_d, dtype=torch.long), torch.tensor(token_type_ids_d, dtype=torch.long), self.domain_idx[idx]

   
class TwinBert(nn.Module):
    def __init__(self, dropout=0.4, encoder='distilbert-base-uncased'):
        super(TwinBert, self).__init__()
        config = RobertaConfig.from_pretrained(encoder)
        config.hidden_dropout_prob = dropout
        self.q_model = BertModel.from_pretrained(encoder, config=config)
        self.k_model = BertModel.from_pretrained(encoder, config=config)
        self.predict_model = nn.Linear(768*2, 2)
        self.sm = nn.Softmax(dim=1)

    def forward_once_query(self, ids, mask, token_type_ids):
        _, output= self.q_model(ids, attention_mask = mask, token_type_ids = token_type_ids)
        # _, output= self.q_model(ids, attention_mask = mask)
        return output
    
    def forward_once_keyword(self, ids, mask, token_type_ids):
        _, output= self.k_model(ids, attention_mask = mask, token_type_ids = token_type_ids)
        # _, output= self.k_model(ids, attention_mask = mask)

        return output

    def forward(self, ids, mask, token_type_ids):
        output1 = self.forward_once_query(ids[0],mask[0], token_type_ids[0])
        output2 = self.forward_once_keyword(ids[1],mask[1], token_type_ids[1])

        logits = self.predict(output1, output2)
        return logits

    def predict(self, query, keyword):
        predict_input = torch.cat([query, keyword], dim=-1)
        logits = self.predict_model(predict_input)
        
        # return self.sm(logits)
        return logits

class GraphTwinBert(nn.Module):
    def __init__(self, dropout=0.4, encoder='distilbert-base-uncased', gnn=None, with_description=False, use_domain_emb=False):
        super(GraphTwinBert, self).__init__()
        config = BertConfig.from_pretrained(encoder)
        config.hidden_dropout_prob = dropout
        self.hidden_dim = config.hidden_size
        self.q_model = BertModel.from_pretrained(encoder, config=config)
        # self.q_model = BertModel(config)
        self.k_model = BertModel.from_pretrained(encoder, config=config)
        self.gnn = gnn
        if self.gnn == 'GAT':
            self.gnnconv = GATConv(self.hidden_dim, int(self.hidden_dim / 3), num_heads=3)
        elif self.gnn == 'GCN':
            self.gnnconv = GraphConv(self.hidden_dim, self.hidden_dim)
        elif self.gnn == 'Sage_mean':
            self.gnnconv = SAGEConv(self.hidden_dim, self.hidden_dim, 'mean')
        elif self.gnn == 'Sage_lstm':
            self.gnnconv = SAGEConv(self.hidden_dim, self.hidden_dim, 'lstm')
        self.dropout = nn.Dropout(p=dropout)
        self.with_description = with_description
        self.use_domain_emb = use_domain_emb
        if self.with_description:
            # self.d_model = BertModel.from_pretrained(encoder, config=config)
            if not self.use_domain_emb:
                self.predict_model = nn.Sequential(
                  nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(self.hidden_dim, 32),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(32, 2)
                )
            else:
                domain_emb = np.load('data/domain_emb.npy')
                self.domain_embedding = nn.Embedding.from_pretrained(torch.from_numpy(domain_emb))
                self.domain_embedding.weight.requires_grad = True
                self.predict_model = nn.Sequential(
                  nn.Linear(self.hidden_dim * 3 + 256, self.hidden_dim),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(self.hidden_dim, 32),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(32, 2)
                )

        else:
            self.d_model = None
            # self.predict_model = nn.Linear(self.hidden_dim * 2, 2)
            if not self.use_domain_emb:
                self.predict_model = nn.Sequential(
                  nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(self.hidden_dim, 32),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(32, 2)
                )
            else:
                domain_emb = np.load('data/domain_emb.npy')
                self.domain_embedding = nn.Embedding.from_pretrained(torch.from_numpy(domain_emb))
                self.domain_embedding.weight.requires_grad = True
                self.predict_model = nn.Sequential(
                  nn.Linear(self.hidden_dim * 2 + 256, self.hidden_dim),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(self.hidden_dim, 32),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(32, 2)
                )
        self.sm = nn.Softmax(dim=1)

    def forward_once_query(self, ids, mask, token_type_ids):
        _, output= self.q_model(ids, attention_mask = mask, token_type_ids = token_type_ids)
        # _, output= self.q_model(ids, attention_mask = mask)
        return output
    
    def forward_once_keyword(self, ids, mask, token_type_ids):
        _, output= self.k_model(ids, attention_mask = mask, token_type_ids = token_type_ids)
        #_, output= self.k_model(ids, attention_mask = mask)

        return output

    def forward_once_domain(self, ids, mask, token_type_ids):
        _, output= self.d_model(ids, attention_mask = mask, token_type_ids = token_type_ids)
        #_, output= self.k_model(ids, attention_mask = mask)

        return output

    def forward(self, graphs, q_ids_neighbor, q_mask_neighbor, q_token_type_neighbor, d_ids_neighbor, d_mask_neighbor, d_token_type_neighbor, domain_idx):
        output_q = self.forward_once_query(q_ids_neighbor, q_mask_neighbor, q_token_type_neighbor)
        output_k = self.forward_once_keyword(graphs.ndata['ids_neighbor'], graphs.ndata['mask_neighbor'], graphs.ndata['token_type_neighbor'])
        if self.gnn is not None:
            # output_k = F.relu(self.gatconv(graphs, output_k).flatten(1))
            output_k = self.gnnconv(graphs, output_k).flatten(1)

        # Pooling the subgraphs
        with graphs.local_scope():
            graphs.ndata['masked_embedding'] = torch.mul(output_k, graphs.ndata['mask_readout'])
            output_k = dgl.sum_nodes(graphs, 'masked_embedding')
        if self.with_description:
            output_d = self.forward_once_query(d_ids_neighbor, d_mask_neighbor, d_token_type_neighbor)
        else:
            output_d = None
        if self.use_domain_emb:
            domain_emb = self.domain_embedding(domain_idx).to(torch.float32)
        else:
            domain_emb = None
        logits = self.predict(output_q, output_k, output_d, domain_emb)
        return logits

    def predict(self, query, keyword, domain, domain_emb):
        if not self.with_description:
            if not self.use_domain_emb:
                predict_input = torch.cat([query, keyword], dim=-1)
            else:
                predict_input = torch.cat([query, keyword, domain_emb], dim=-1)
        else:
            if not self.use_domain_emb:
                predict_input = torch.cat([query, keyword, domain], dim=-1)
            else:
                predict_input = torch.cat([query, keyword, domain, domain_emb], dim=-1)
        logits = self.predict_model(predict_input)
        
        return logits


class TokenModel(nn.Module):
    def __init__(self, dropout=0.4, encoder='bert-base-uncased', with_description=False, use_domain_emb=False):
        super(TokenModel, self).__init__()
        config = BertConfig.from_pretrained(encoder)
        config.hidden_dropout_prob = dropout
        self.hidden_dim = config.hidden_size
        self.q_model = BertModel.from_pretrained(encoder, config=config)
        self.k_model = GraphBertModel.from_pretrained(encoder, config=config)
        self.with_description = with_description
        self.use_domain_emb = use_domain_emb
        if self.with_description:
            self.d_model = BertModel.from_pretrained(encoder, config=config)
            if not self.use_domain_emb:
                self.predict_model = nn.Sequential(
                  nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(self.hidden_dim, 32),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(32, 2)
                )
            else:
                domain_emb = np.load('data/domain_emb.npy')
                self.domain_embedding = nn.Embedding.from_pretrained(torch.from_numpy(domain_emb))
                self.domain_embedding.weight.requires_grad = True
                self.predict_model = nn.Sequential(
                  nn.Linear(self.hidden_dim * 3 + 256, self.hidden_dim),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(self.hidden_dim, 32),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(32, 2)
                )

        else:
            self.d_model = None
            # self.predict_model = nn.Linear(self.hidden_dim * 2, 2)
            if not self.use_domain_emb:
                self.predict_model = nn.Sequential(
                  nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(self.hidden_dim, 32),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(32, 2)
                )
            else:
                domain_emb = np.load('data/domain_emb.npy')
                self.domain_embedding = nn.Embedding.from_pretrained(torch.from_numpy(domain_emb))
                self.domain_embedding.weight.requires_grad = True
                self.predict_model = nn.Sequential(
                  nn.Linear(self.hidden_dim * 2 + 256, self.hidden_dim),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(self.hidden_dim, 32),
                  nn.Dropout(dropout),
                  nn.ReLU(),
                  nn.Linear(32, 2)
                )
        self.sm = nn.Softmax(dim=1)
    
    def forward_once_query(self, ids, mask, token_type_ids):
        _, output= self.q_model(ids, attention_mask = mask, token_type_ids = token_type_ids)
        # _, output= self.q_model(ids, attention_mask = mask)
        return output
    
    def forward_once_keyword(self, ids, mask, token_type_ids, graph):
        _, output= self.k_model(ids, attention_mask = mask, token_type_ids = token_type_ids, graph=graph)
        #_, output= self.k_model(ids, attention_mask = mask)

        return output

    def forward_once_domain(self, ids, mask, token_type_ids):
        _, output= self.d_model(ids, attention_mask = mask, token_type_ids = token_type_ids)
        #_, output= self.k_model(ids, attention_mask = mask)

        return output
    
    def forward(self, graphs, q_ids_neighbor, q_mask_neighbor, q_token_type_neighbor, d_ids_neighbor, d_mask_neighbor, d_token_type_neighbor, domain_idx):
        output_q = self.forward_once_query(q_ids_neighbor, q_mask_neighbor, q_token_type_neighbor)
        output_k = self.forward_once_keyword(graphs.ndata['ids_neighbor'], graphs.ndata['mask_neighbor'], graphs.ndata['token_type_neighbor'], graphs)

        # Pooling the subgraphs
        with graphs.local_scope():
            graphs.ndata['masked_embedding'] = torch.mul(output_k, graphs.ndata['mask_readout'])
            output_k = dgl.sum_nodes(graphs, 'masked_embedding')
        if self.with_description:
            output_d = self.forward_once_domain(d_ids_neighbor, d_mask_neighbor, d_token_type_neighbor)
        else:
            output_d = None
        if self.use_domain_emb:
            domain_emb = self.domain_embedding(domain_idx).to(torch.float32)
        else:
            domain_emb = None
        logits = self.predict(output_q, output_k, output_d, domain_emb)
        return logits

    def predict(self, query, keyword, domain, domain_emb):
        if not self.with_description:
            if not self.use_domain_emb:
                predict_input = torch.cat([query, keyword], dim=-1)
            else:
                predict_input = torch.cat([query, keyword, domain_emb], dim=-1)
        else:
            if not self.use_domain_emb:
                predict_input = torch.cat([query, keyword, domain], dim=-1)
            else:
                predict_input = torch.cat([query, keyword, domain, domain_emb], dim=-1)
        logits = self.predict_model(predict_input)
        return logits


class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0.4):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cos_sim = F.cosine_similarity(output1, output2)
        loss_cos_con = torch.mean((1-label) * torch.div(torch.pow((1.0-cos_sim), 2), 4) +
                                    (label) * torch.pow(cos_sim * torch.lt(cos_sim, self.margin), 2))
        return loss_cos_con

class CosineCrossEntropyLoss(nn.Module):
    def __init__(self, margin=0.4):
        super(CosineCrossEntropyLoss, self).__init__()
    
    def forward(self, output1, output2, label):
        cos_sim = torch.sigmoid(F.cosine_similarity(output1, output2))
        loss_cross_entropy = -torch.mean((1-label) * torch.log(1 - cos_sim) + label * torch.log(cos_sim))
        return loss_cross_entropy

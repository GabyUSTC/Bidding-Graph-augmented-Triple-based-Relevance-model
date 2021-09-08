import pandas as pd
import os
import dgl
import numpy as np
import random
import torch

def load_data(dataset, **kwargs):
    if dataset == 'bing_domain':
        train_dataset = pd.read_csv("adaptive_weights/train.tsv", sep='\t') 
        test_dataset = pd.read_csv("adaptive_weights/dev.tsv", sep='\t') 

        return train_dataset[['query', 'keyword', 'label']], test_dataset[['query', 'keyword', 'label']]
    elif dataset == 'bing_co_order':
        if kwargs['type'] == 'topk':
            topk = kwargs['topk']

            train_file = "adaptive_weights/train_" + "top_" + str(topk) + ".tsv"
            test_file = "adaptive_weights/test_" + "top_" + str(topk) + ".tsv"

            # train_file = "new_data/train.tsv"
            # test_file = "new_data/dev.tsv"
            if os.path.exists(train_file):
                train_dataset = pd.read_csv(train_file, sep='\t') 
            else:
                train_dataset = pd.read_csv("adaptive_weights/train.tsv", sep='\t') 
                for i in range(train_dataset.shape[0]):
                    head, sep, tail = train_dataset['keyword'][i].partition(' [SEP] ')
                    train_dataset['keyword'][i] = head
                    if type(train_dataset['co-order'][i]) == str:
                        co_orders = train_dataset['co-order'][i].split(' ##! ')
                        for j, co_order_key in enumerate(co_orders):
                            co_orders[j] = co_orders[j].split(' @_% ')
                            co_orders[j][1] = float(co_orders[j][1])
                        co_orders = sorted(co_orders, key=lambda x:x[1], reverse=True)
                        co_order_words = [i[0] for i in co_orders]
                        # co_order_words_de_dup = list(set(co_order_words))
                        # co_order_words_de_dup.sort(key=co_order_words.index)
                        last_word = ''
                        count = 0
                        for k in range(len(co_order_words)):
                            if co_order_words[k] != last_word:
                                train_dataset['keyword'][i] = train_dataset['keyword'][i] + ' ##! ' + co_order_words[k]
                                count += 1
                                last_word = co_order_words[k]
                            if count == topk:
                                break
                print('train set processing done')

            if os.path.exists(test_file):
                test_dataset = pd.read_csv(test_file, sep='\t') 
            else:
                test_dataset = pd.read_csv("adaptive_weights/dev.tsv", sep='\t') 
                for i in range(test_dataset.shape[0]):
                    head, sep, tail = test_dataset['keyword'][i].partition(' [SEP] ')
                    test_dataset['keyword'][i] = head
                    if type(test_dataset['co-order'][i]) == str:
                        co_orders = test_dataset['co-order'][i].split(' ##! ')
                        for j, co_order_key in enumerate(co_orders):
                            co_orders[j] = co_orders[j].split(' @_% ')
                            co_orders[j][1] = float(co_orders[j][1])
                        co_orders = sorted(co_orders, key=lambda x:x[1], reverse=True)
                        co_order_words = [i[0] for i in co_orders]
                        # co_order_words_de_dup = list(set(co_order_words))
                        # co_order_words_de_dup.sort(key=co_order_words.index)
                        last_word = ''
                        count = 0
                        for k in range(len(co_order_words)):
                            if co_order_words[k] != last_word:
                                test_dataset['keyword'][i] = test_dataset['keyword'][i] + ' ##! ' + co_order_words[k]
                                count += 1
                                last_word = co_order_words[k]
                            if count == topk:
                                break
                print('test set processing done')
                
                with open(train_file, 'w') as train_writer:
                    train_writer.write(train_dataset[['query', 'keyword', 'label']].to_csv(sep='\t', index=False))
                
                with open(test_file, 'w') as test_writer:
                    test_writer.write(test_dataset[['query', 'keyword', 'label']].to_csv(sep='\t', index=False))

        elif kwargs['type'] == 'threshold':
            threshold = kwargs['threshold']
        
        # return train_dataset[:1024], test_dataset[:1024]
        return train_dataset, test_dataset

def load_data_new(dataset, **kwargs):
    if dataset == 'bing_domain':
        train_dataset = pd.read_csv("adaptive_weights/train.tsv", sep='\t') 
        test_dataset = pd.read_csv("adaptive_weights/dev.tsv", sep='\t') 

        return train_dataset[['query', 'keyword', 'label']], test_dataset[['query', 'keyword', 'label']]
    elif dataset == 'bing_co_order':
        if kwargs['type'] == 'topk':
            topk = kwargs['topk']

            train_file = "processed_data/train_" + "top_" + str(topk) + ".tsv"
            test_file = "processed_data/test_" + "top_" + str(topk) + ".tsv"

            if os.path.exists(train_file):
                train_dataset = pd.read_csv(train_file, sep='\t') 
            else:
                train_dataset = pd.read_csv("processed_data/train.tsv", sep='\t') 
                for i in range(train_dataset.shape[0]):
                    head, sep, tail = train_dataset['keyword'][i].partition(' [SEP] ')
                    train_dataset['keyword'][i] = head
                    if type(train_dataset['co-order'][i]) == str:
                        co_orders = train_dataset['co-order'][i].split(' ##! ')
                        if len(co_orders) >= topk:
                            co_orders = co_orders[: topk]
                        neighbors = head
                        for neighbor in co_orders:
                            neighbors = neighbors + ' ##! ' + neighbor
                        train_dataset['keyword'][i] = neighbors
                    else:
                        train_dataset['keyword'][i] = head
                print('train set processing done')

            if os.path.exists(test_file):
                test_dataset = pd.read_csv(test_file, sep='\t') 
            else:
                test_dataset = pd.read_csv("processed_data/dev.tsv", sep='\t') 
                for i in range(test_dataset.shape[0]):
                    head, sep, tail = test_dataset['keyword'][i].partition(' [SEP] ')
                    test_dataset['keyword'][i] = head
                    if type(test_dataset['co-order'][i]) == str:
                        co_orders = test_dataset['co-order'][i].split(' ##! ')
                        if len(co_orders) >= topk:
                            co_orders = co_orders[: topk]
                        neighbors = head
                        for neighbor in co_orders:
                            neighbors = neighbors + ' ##! ' + neighbor
                        test_dataset['keyword'][i] = neighbors
                    else:
                        test_dataset['keyword'][i] = head
                print('test set processing done')
                
                with open(train_file, 'w') as train_writer:
                    train_writer.write(train_dataset[['query', 'keyword', 'label']].to_csv(sep='\t', index=False))
                
                with open(test_file, 'w') as test_writer:
                    test_writer.write(test_dataset[['query', 'keyword', 'label']].to_csv(sep='\t', index=False))

        elif kwargs['type'] == 'randk':
            randk = kwargs['randk']

            train_file = "processed_data/train_" + "rand_" + str(randk) + ".tsv"
            test_file = "processed_data/test_" + "rand_" + str(randk) + ".tsv"

            if os.path.exists(train_file):
                train_dataset = pd.read_csv(train_file, sep='\t') 
            else:
                train_dataset = pd.read_csv("processed_data/train.tsv", sep='\t') 
                for i in range(train_dataset.shape[0]):
                    head, sep, tail = train_dataset['keyword'][i].partition(' [SEP] ')
                    train_dataset['keyword'][i] = head
                    if type(train_dataset['co-order'][i]) == str:
                        co_orders = train_dataset['co-order'][i].split(' ##! ')
                        if len(co_orders) >= randk:
                            co_orders = random.sample(co_orders, randk)
                        neighbors = head
                        for neighbor in co_orders:
                            neighbors = neighbors + ' ##! ' + neighbor
                        train_dataset['keyword'][i] = neighbors
                    else:
                        train_dataset['keyword'][i] = head
                print('train set processing done')

            if os.path.exists(test_file):
                test_dataset = pd.read_csv(test_file, sep='\t') 
            else:
                test_dataset = pd.read_csv("processed_data/dev.tsv", sep='\t') 
                for i in range(test_dataset.shape[0]):
                    head, sep, tail = test_dataset['keyword'][i].partition(' [SEP] ')
                    test_dataset['keyword'][i] = head
                    if type(test_dataset['co-order'][i]) == str:
                        co_orders = test_dataset['co-order'][i].split(' ##! ')
                        if len(co_orders) >= randk:
                            co_orders = random.sample(co_orders, randk)
                        neighbors = head
                        for neighbor in co_orders:
                            neighbors = neighbors + ' ##! ' + neighbor
                        test_dataset['keyword'][i] = neighbors
                    else:
                        test_dataset['keyword'][i] = head
                print('test set processing done')
                
                with open(train_file, 'w') as train_writer:
                    train_writer.write(train_dataset[['query', 'keyword', 'label']].to_csv(sep='\t', index=False))
                
                with open(test_file, 'w') as test_writer:
                    test_writer.write(test_dataset[['query', 'keyword', 'label']].to_csv(sep='\t', index=False))
        
        # return train_dataset[:1024], test_dataset[:1024]
        return train_dataset, test_dataset

def load_data_large(dataset, **kwargs):
    if dataset == 'bing_domain':
        train_dataset = pd.read_csv("new_data/train.tsv", sep='\t') 
        val_dataset = pd.read_csv("new_data/val.tsv", sep='\t') 
        test_dataset = pd.read_csv("new_data/test.tsv", sep='\t') 

        return train_dataset[['query', 'keyword', 'label']], test_dataset[['query', 'keyword', 'label']]
    elif dataset == 'bing_co_order':
        if kwargs['type'] == 'topk':
            topk = kwargs['topk']

            train_file = "new_data/train_" + "top_" + str(topk) + ".tsv"
            val_file = "new_data/val_" + "top_" + str(topk) + ".tsv"
            test_file = "new_data/test_" + "top_" + str(topk) + ".tsv"

            if os.path.exists(train_file):
                train_dataset = pd.read_csv(train_file, sep='\t') 
                if 'description' not in train_dataset.keys():
                    train_dataset_with_desc = pd.read_csv('data/train.tsv', sep='\t') 
                    train_dataset['description'] = train_dataset_with_desc['description']
                    with open(train_file, 'w') as train_writer:
                        train_writer.write(train_dataset[['query', 'keyword', 'label', 'description']].to_csv(sep='\t', index=False))
            else:
                train_dataset = pd.read_csv("data/train.tsv", sep='\t') 
                for i in range(train_dataset.shape[0]):
                    head, sep, tail = train_dataset['keyword'][i].partition(' [SEP] ')
                    train_dataset['keyword'][i] = head
                    if type(train_dataset['co-order'][i]) == str:
                        co_orders = train_dataset['co-order'][i].split(' ##! ')
                        if len(co_orders) >= topk:
                            co_orders = co_orders[: topk]
                        neighbors = head
                        for neighbor in co_orders:
                            neighbors = neighbors + ' ##! ' + neighbor
                        train_dataset['keyword'][i] = neighbors
                    else:
                        train_dataset['keyword'][i] = head
                print('train set processing done')
            
            if os.path.exists(val_file):
                val_dataset = pd.read_csv(val_file, sep='\t') 
                if 'description' not in val_dataset.keys():
                    val_dataset_with_desc = pd.read_csv('data/valid.tsv', sep='\t') 
                    val_dataset['description'] = val_dataset_with_desc['description']
                    with open(val_file, 'w') as val_writer:
                        val_writer.write(val_dataset[['query', 'keyword', 'label', 'description']].to_csv(sep='\t', index=False))
            else:
                val_dataset = pd.read_csv("data/valid.tsv", sep='\t') 
                for i in range(val_dataset.shape[0]):
                    head, sep, tail = val_dataset['keyword'][i].partition(' [SEP] ')
                    val_dataset['keyword'][i] = head
                    if type(val_dataset['co-order'][i]) == str:
                        co_orders = val_dataset['co-order'][i].split(' ##! ')
                        if len(co_orders) >= topk:
                            co_orders = co_orders[: topk]
                        neighbors = head
                        for neighbor in co_orders:
                            neighbors = neighbors + ' ##! ' + neighbor
                        val_dataset['keyword'][i] = neighbors
                    else:
                        val_dataset['keyword'][i] = head
                print('val set processing done')

            if os.path.exists(test_file):
                test_dataset = pd.read_csv(test_file, sep='\t') 
                if 'description' not in test_dataset.keys():
                    test_dataset_with_desc = pd.read_csv('data/test.tsv', sep='\t') 
                    test_dataset['description'] = test_dataset_with_desc['description']
                    with open(test_file, 'w') as test_writer:
                        test_writer.write(test_dataset[['query', 'keyword', 'label', 'description']].to_csv(sep='\t', index=False))
            else:
                test_dataset = pd.read_csv("data/test.tsv", sep='\t') 
                for i in range(test_dataset.shape[0]):
                    head, sep, tail = test_dataset['keyword'][i].partition(' [SEP] ')
                    test_dataset['keyword'][i] = head
                    if type(test_dataset['co-order'][i]) == str:
                        co_orders = test_dataset['co-order'][i].split(' ##! ')
                        if len(co_orders) >= topk:
                            co_orders = co_orders[: topk]
                        neighbors = head
                        for neighbor in co_orders:
                            neighbors = neighbors + ' ##! ' + neighbor
                        test_dataset['keyword'][i] = neighbors
                    else:
                        test_dataset['keyword'][i] = head
                print('test set processing done')
                
                with open(train_file, 'w') as train_writer:
                    train_writer.write(train_dataset[['query', 'keyword', 'label', 'description']].to_csv(sep='\t', index=False))
                
                with open(val_file, 'w') as val_writer:
                    val_writer.write(val_dataset[['query', 'keyword', 'label', 'description']].to_csv(sep='\t', index=False))
                
                with open(test_file, 'w') as test_writer:
                    test_writer.write(test_dataset[['query', 'keyword', 'label', 'description']].to_csv(sep='\t', index=False))
        elif kwargs['type'] == 'randk':
            randk = kwargs['randk']

            train_file = "new_data/train_" + "rand_" + str(randk) + ".tsv"
            val_file = "new_data/val_" + "rand_" + str(randk) + ".tsv"
            test_file = "new_data/test_" + "rand_" + str(randk) + ".tsv"

            if os.path.exists(train_file):
                train_dataset = pd.read_csv(train_file, sep='\t') 
                if 'description' not in train_dataset.keys():
                    train_dataset_with_desc = pd.read_csv('data/train.tsv', sep='\t') 
                    train_dataset['description'] = train_dataset_with_desc['description']
                    with open(train_file, 'w') as train_writer:
                        train_writer.write(train_dataset[['query', 'keyword', 'label', 'description']].to_csv(sep='\t', index=False))
            else:
                train_dataset = pd.read_csv("data/train.tsv", sep='\t')  
                for i in range(train_dataset.shape[0]):
                    head, sep, tail = train_dataset['keyword'][i].partition(' [SEP] ')
                    train_dataset['keyword'][i] = head
                    if type(train_dataset['co-order'][i]) == str:
                        co_orders = train_dataset['co-order'][i].split(' ##! ')
                        if len(co_orders) >= randk:
                            co_orders = random.sample(co_orders, randk)
                        neighbors = head
                        for neighbor in co_orders:
                            neighbors = neighbors + ' ##! ' + neighbor
                        train_dataset['keyword'][i] = neighbors
                    else:
                        train_dataset['keyword'][i] = head
                with open(train_file, 'w') as train_writer:
                    train_writer.write(train_dataset[['query', 'keyword', 'label', 'description']].to_csv(sep='\t', index=False))
                print('train set processing done')
            
            if os.path.exists(val_file):
                val_dataset = pd.read_csv(val_file, sep='\t')
                if 'description' not in val_dataset.keys():
                    val_dataset_with_desc = pd.read_csv('data/valid.tsv', sep='\t') 
                    val_dataset['description'] = val_dataset_with_desc['description']
                    with open(val_file, 'w') as val_writer:
                        val_writer.write(val_dataset[['query', 'keyword', 'label', 'description']].to_csv(sep='\t', index=False)) 
            else:
                val_dataset = pd.read_csv("data/valid.tsv", sep='\t') 
                for i in range(val_dataset.shape[0]):
                    head, sep, tail = val_dataset['keyword'][i].partition(' [SEP] ')
                    val_dataset['keyword'][i] = head
                    if type(val_dataset['co-order'][i]) == str:
                        co_orders = val_dataset['co-order'][i].split(' ##! ')
                        if len(co_orders) >= randk:
                            co_orders = random.sample(co_orders, randk)
                        neighbors = head
                        for neighbor in co_orders:
                            neighbors = neighbors + ' ##! ' + neighbor
                        val_dataset['keyword'][i] = neighbors
                    else:
                        val_dataset['keyword'][i] = head
                with open(val_file, 'w') as val_writer:
                    val_writer.write(val_dataset[['query', 'keyword', 'label', 'description']].to_csv(sep='\t', index=False))
                print('val set processing done')

            if os.path.exists(test_file):
                test_dataset = pd.read_csv(test_file, sep='\t') 
                if 'description' not in test_dataset.keys():
                    test_dataset_with_desc = pd.read_csv('data/test.tsv', sep='\t') 
                    test_dataset['description'] = test_dataset_with_desc['description']
                    with open(test_file, 'w') as test_writer:
                        test_writer.write(test_dataset[['query', 'keyword', 'label', 'description']].to_csv(sep='\t', index=False))
            else:
                test_dataset = pd.read_csv("data/test.tsv", sep='\t') 
                for i in range(test_dataset.shape[0]):
                    head, sep, tail = test_dataset['keyword'][i].partition(' [SEP] ')
                    test_dataset['keyword'][i] = head
                    if type(test_dataset['co-order'][i]) == str:
                        co_orders = test_dataset['co-order'][i].split(' ##! ')
                        if len(co_orders) >= randk:
                            co_orders = random.sample(co_orders, randk)
                        neighbors = head
                        for neighbor in co_orders:
                            neighbors = neighbors + ' ##! ' + neighbor
                        test_dataset['keyword'][i] = neighbors
                    else:
                        test_dataset['keyword'][i] = head
                with open(test_file, 'w') as test_writer:
                    test_writer.write(test_dataset[['query', 'keyword', 'label', 'description']].to_csv(sep='\t', index=False))
                print('test set processing done')
        
        # return train_dataset[:1024], val_dataset[:1024], test_dataset[:1024]
        return train_dataset, val_dataset, test_dataset

def create_graph(dataset):
    '''
    For creating subgraphs centered by keyword when we choose to use GNN models
    '''
    key = [x.split(' ##! ') for x in dataset['keyword']]
    keyword = [x[0] for x in key]

    src, dest, graph = [], [], []
    for n in key:
        src_n = [0 for _ in range(len(n))]
        dest_n = [i for i in range(0, len(n))]
        src.append(src_n)
        dest.append(dest_n)
        graph_n = dgl.graph((src_n, dest_n))
        graph_n = dgl.to_bidirected(graph_n)
        graph.append(graph_n)


    print("------------------Done creating graphs!----------------")

    return {
            'query': list(dataset['query']),
            'keyword': keyword,
            'neighbors': key,
            'src': src,
            'dest': dest,
            'graph': graph,
            'label': list(dataset['label']),
            'description': list(dataset['description']),
            'domain_idx': list(dataset['domain_idx'])
        }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)
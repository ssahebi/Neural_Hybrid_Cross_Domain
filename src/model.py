import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import pdb

class CrossDomainModel(nn.Module):
    """It is the baseline"""
    def __init__(self, config):
        super(CrossDomainModel, self).__init__()
        self.number_users = config['num_users']
        self.number_domains = config['num_domains']

        # it should be a list of integer
        self.number_items_in_domain = config['num_items_in_domain']
        
        self.u_latent_dims = config['u_latent_dim']
        self.i_latent_dims = config['i_latent_dim']

        self.emb_users = nn.Embedding(self.number_users, self.u_latent_dims).to(config['device'])
        self.emb_items = list()
        for i in range(self.number_domains):
            emb_items_in_domain = nn.Embedding(self.number_items_in_domain[i],\
                    self.i_latent_dims).to(config['device'])
            self.emb_items.append(emb_items_in_domain)
        #self.emb_items = nn.ModuleList([nn.Embedding(self.number_items_in_domain[i],
        #    self.i_latent_dims) for i in range(self.number_domains)])

        #
        self.fc_layers = nn.ModuleList()
        layers = config['layers'] # list of integer
        self.number_layers = len(layers)
        for i in range(self.number_domains):
            each_domain = nn.ModuleList()
            self.fc_layers.append(each_domain)
            dim_of_concat = self.u_latent_dims + self.i_latent_dims
            each_domain.append(nn.Linear(dim_of_concat, layers[0]))
            for idx, (in_size, out_size) in \
                    enumerate(zip(layers[:-1], layers[1:])):
                each_domain.append(nn.Linear(in_size, out_size))
        self.affine_output = nn.Linear(in_features=layers[-1], out_features=1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_idx, item_idx, d_idx):
        """user_idx : index of user
           item_idxs: list of (domain id, item id in this domain)
        """
        #print(str(user_idx) + ':' + str(item_idx) + ':' + str(d_idx))
        u_emb = self.emb_users(user_idx)
        v_emb = self.emb_items[d_idx](item_idx)
        vector = torch.cat([u_emb, v_emb], 1)
        for idx, _ in enumerate(range(self.number_layers)):
            #print(str(idx) + ':' + str(self.number_layers))
            vector = self.fc_layers[d_idx][idx](vector)
            vector = torch.nn.ReLU()(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits) * 5
        return rating

class AutoGenReview(CrossDomainModel):
    def __init__(self, config):
        super(AutoGenReview, self).__init__(config)
        self.vocab_size = config['vocab_size']
        self.vocab_emb_size = config['vocab_emb_size']
        self.lstm_hidden_size = config['lstm_hidden_size']
        self.lstm_num_layers = config['lstm_num_layers']

        self.word_emb = nn.Embedding(self.vocab_size, self.vocab_emb_size).to(config['device'])
        lstm_input_size = self.vocab_emb_size + self.u_latent_dims + self.i_latent_dims # since we concat embedding of 
        self.lstm = nn.LSTM(lstm_input_size, self.lstm_hidden_size, self.lstm_num_layers)
        self.linear = nn.Linear(self.lstm_hidden_size, self.vocab_size)
        self.max_seg_length = config.get('max_seg_length', 20)

    def forward(self, u_idx, i_idx, d_idx, review, length):
        print(u_idx)
        u_emb = self.emb_users(u_idx).squeeze()
        v_emb = None
        for j in range(len(d_idx)):
            if v_emb is None:
                emb = self.emb_items[d_idx[j]]
                v_emb = emb(i_idx[j])
            else:
                new_v_emb = self.emb_items[d_idx[j]](i_idx[j])
                v_emb = torch.cat([v_emb, new_v_emb], dim = 0)
        vector = torch.cat([u_emb, v_emb], 1)
        uv_emb_concat = vector
        for idx, _ in enumerate(range(self.number_layers)):
            out_vector = None
            for j in range(len(d_idx)):
                #print(str(idx) + ':' + str(self.number_layers))
                #vector = self.fc_layers[d_idx][idx](vector)
                #vector = torch.nn.ReLU()(vector)
                layer = self.fc_layers[d_idx[j]][idx]
                if out_vector is None:
                    out_vector = layer(vector[j]).unsqueeze(0)
                else:
                    new_out_vector = layer(vector[j])
                    out_vector = torch.cat([out_vector, new_out_vector.unsqueeze(0)], dim=0)
            out_vector = torch.nn.ReLU()(out_vector)
            vector = out_vector

        logits = self.affine_output(vector)
        rating = self.logistic(logits) * 5

        # for review
        #outputs = None
        #if review != None:
        w_embedding = self.word_emb(review)
        #print(w_embedding.size())
        #print(vector.size())
        #print(vector.unsqueeze(1).size())
        v = uv_emb_concat.unsqueeze(1)
        repeat_vals = [-1] +  [w_embedding.shape[1] // v.shape[1]] + [-1]
        embedding = torch.cat([v.expand(*repeat_vals), w_embedding], dim=2)

        #embedding = torch.cat((vector.unsqueeze(1), w_embedding), 2)
        packed = pack_padded_sequence(embedding, length, batch_first=True)
        #print(packed.size())
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return rating, outputs

    def sample(self, u_idx, i_idx):
        # generate review of user and item
        pass

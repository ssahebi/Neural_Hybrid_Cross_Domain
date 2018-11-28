import torch, pickle
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from model import CrossDomainModel, AutoGenReview
from data_loader import get_review_loader, ReviewDataset
from build_vocab import Vocabulary
from build_user_item import RatingReviewInfo

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    config = {'num_domains': 2, 'num_items_in_domain': [4, 3],\
            'u_latent_dim': 8, 'i_latent_dim': 8, 'vocab_emb_size': 5,
            'lstm_hidden_size': 512, 'lstm_num_layers': 1,
            'layers': [16,64,32,16,8],
            'learning_rate': 0.0001}
    # load vocabulary
    with open('../data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    config['vocab_size'] = len(vocab)

    # load user item
    with open('../data/user_item.pkl', 'rb') as f:
        uiData = pickle.load(f)
    config['num_users'] = uiData.getNumUsers()
    config['num_items_in_domain'] = uiData.getNumItems()

    # build data loader
    info = {'num_domain': 2, 
            'data': [('../data/ratings_Musical_Instruments.csv', 
                      '../data/reviews_Musical_Instruments_5.json'),
                     ('../data/ratings_Amazon_Instant_Video.csv', 
                      '../data/reviews_Amazon_Instant_Video_5.json')]}
    loader = get_review_loader(info, vocab, uiData)
    
    # build models
    model = AutoGenReview(config).to(device)

    # loss and optimization
    cri_review = nn.CrossEntropyLoss() # criterion for review
    cri_rating = nn.MSELoss()  # criterion for rating
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=config['learning_rate'])

    # Train the model
    for epoch in range(10): # TODO: tune the number of epoch 
        for (u, i, d, rating, review, lengths) in loader:
            u = u.to(device)
            i = i.to(device)
            targets = pack_padded_sequence(review, lengths, batch_first=True)[0]

            predict_rating, output = model(u, i, d, review, lengths)
            #rating_loss = cri_rating(rating, predict_rating)
            rating_loss = cri_rating(predict_rating, rating)
            review_loss = cri_review(output, targets)
            loss = rating_loss + review_loss
            
            model.zero_grad()
            loss.backward()
            optimizer.step()

def test():
    #data = torch.utils.data.DataLoader()
    # user id, domain id, item id, rating
    data = [(0, 0, 0, 1),
            (0, 0, 2, 5),
            (1, 0, 3, 5),
            (2, 0, 2, 2),
            (3, 0, 0, 4),
            (3, 0, 3, 2),
            (4, 0, 2, 6),
            (0, 1, 0, 1),
            (0, 1, 2, 2),
            (1, 1, 1, 5),
            (2, 1, 0, 4),
            (3, 1, 1, 2),
            (4, 1, 2, 1)]
    config = {'num_users': 5, 'num_domains': 2, 'num_items_in_domain': [4, 3],\
            'u_latent_dim': 8, 'i_latent_dim': 8, 'layers': [16,64,32,16,8]}

    model = CrossDomainModel(config).to(device)

    # loss and optimization
    criterion = nn.MSELoss()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params) #, lr=args.learning_rate)

    # Train the model
    for epoch in range(100): # run with 10 epoch
        for i, (u_idx, d_idx, i_idx, target) in enumerate(data):
            u  = torch.LongTensor([u_idx])#, dtype=torch.long)
            ii = torch.LongTensor([i_idx])#, dtype=torch.long) 
            ratings = model(u, ii, d_idx)
            loss = criterion(ratings, torch.Tensor([[target]]))
            model.zero_grad()
            loss.backward()
            optimizer.step()
            print(d_idx)
            print(loss.item())
            print('--------------')
    print(torch.mm(model.emb_users.weight.data, torch.t(model.emb_items[0].weight.data)))
    print(torch.mm(model.emb_users.weight.data, torch.t(model.emb_items[1].weight.data)))

if __name__ == '__main__':
    #test()
    main()

from torch.utils.data import Dataset, DataLoader
import json, nltk, torch

class ReviewDataset(Dataset):
    def __init__(self, info, vocab, uiData): # TODO: need to add parameters
        self.vocab = vocab
        self.num_domain = info['num_domain']
        self.data = list()
        for d_idx in range(info['num_domain']): 
            # tuple: (check-in file, review file)
            rating_fname, review_fname = info['data'][d_idx]

            rating_data = list()
            # read rating
            with open(rating_fname, 'r') as f:
                lines = f.readlines()
            for line in lines:
                comp = line.split(',')
                uId = comp[0]
                if not uiData.hasUser(uId):
                    continue
                vId = comp[1]
                rating = float(comp[2])
                cont_uId = uiData.call_user(uId)
                cont_vId = uiData.call_item(vId, d_idx)
                rating_data.append((cont_uId, cont_vId, rating)) 

            # read review
            review_data = dict()
            with open(review_fname, 'r') as f:
                lines = f.readlines()
            for line in lines:
                comp = json.loads(line)
                if not uiData.hasUser(comp['reviewerID']):
                    continue
                uId = uiData.call_user(comp['reviewerID'])
                iId = uiData.call_item(comp['asin'], d_idx)
                text = comp['reviewText']
                re = review_data.get(uId)
                if re == None:
                    re = dict()
                    review_data[uId] = re
                re[iId] = text

            # merge rating and review data
            for uId, iId, rating in rating_data:
                text = None
                reOfUsers = review_data.get(uId)
                if reOfUsers != None:
                    text = reOfUsers.get(iId) 
                self.data.append((uId, iId, d_idx, rating, text))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
            return a tuple (user id, item id, rating, review)
            review has been converted to word ids.
        """

        # convert review to word ids
        datapoint = self.data[idx]

        tokens = nltk.tokenize.word_tokenize(str(datapoint[4]).lower())
        review = []
        review.append(self.vocab('<start>'))
        review.extend([self.vocab(token) for token in tokens])
        review.append(self.vocab('<end>'))
        target = torch.Tensor(review)
        
        tu = torch.LongTensor([datapoint[0]])
        ti = torch.LongTensor([datapoint[1]])
        
        d_idx = torch.LongTensor([datapoint[2]])

        return tu, ti, datapoint[2], torch.FloatTensor([datapoint[3]]), target
        #return tu, ti, d_idx, datapoint[3], target

class RatingDataset(Dataset):
    def __init__(self, info):
        pass
    def __len__(self):
        pass # TODO
    def __getitem__(self, idx):
        pass

def collate_fn(data):
    """
        Since the length of reviews are vary, we need to make them equal
    """
    # Sort a data list by review length (descending order).
    data.sort(key=lambda x: len(x[4]), reverse=True)
    tu, ti, d, rating, reviews = zip(*data)

    tu = torch.stack(tu, 0) # merge user id
    ti = torch.stack(ti, 0) # merge item id
    dd = [i[2] for i in data]
    #dd = torch.stack(dd, 0)
    #rr = [i[3] for i in data]
    rr = torch.stack(rating, 0)

    # merge review
    lengths = [len(cap) for cap in reviews]
    targets = torch.zeros(len(reviews), max(lengths)).long()
    for i, cap in enumerate(reviews):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return tu, ti, dd, rr, targets, lengths

def get_review_loader(info, vocab, uiData, batch_size=10, shuffle=True, 
        num_workers=10):
    # TODO put the parameter here
    data = ReviewDataset(info, vocab, uiData)
    dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, num_workers= num_workers, collate_fn=collate_fn)
    return dataloader

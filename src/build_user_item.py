import pickle, glob

class RatingReviewInfo(object):
    def __init__(self, K):
        self.K = K # it is the number of domain
        self.user2idx = dict()
        self.idx2user = dict()

        self.item2idx = list()
        self.idx2item = list()

        self.u_idx = 0 # since we share users across multiple domains
        self.i_idx = list()
        for i in range(K):
            self.i_idx.append(0)
            self.item2idx.append(dict())
            self.idx2item.append(dict())

    def add_user(self, userId):
        """
            userId: the real id of user
        """
        if not userId in self.user2idx:
            self.user2idx[userId] = self.u_idx
            self.idx2user[self.u_idx] = userId
            self.u_idx += 1

    def call_user(self, userId):
        return self.user2idx.get(userId)

    def add_item(self, itemId, domainId):
        """
            itemId: the real id of item
            domainId: the id of domain
        """
        if not itemId in self.item2idx[domainId]:
            self.item2idx[domainId][itemId] = self.i_idx[domainId]
            self.idx2item[domainId][self.i_idx[domainId]] = itemId
            self.i_idx[domainId] += 1

    def call_item(self, itemId, domainId):
        return self.item2idx[domainId].get(itemId)
    
    def getNumUsers(self):
        return len(self.user2idx)

    def getNumItems(self):
        result = []
        for i in range(self.K):
            result.append(len(self.item2idx[i]))
        return result

    def hasUser(self, uId):
        result = uId in self.user2idx
        return result

def build_info(rating_files):
    userAllDomains = list()
    for fname in rating_files:
        userEachDomain = set()
        with open(fname, 'r') as f:
            lines = f.readlines()
        for line in lines:
            comp = line.split(',')
            uId = comp[0]
            userEachDomain.add(uId)
        userAllDomains.append(userEachDomain)
    # users who rating in all domains
    intersection_users = set.intersection(*userAllDomains)

    info = RatingReviewInfo(len(rating_files))
    for u in intersection_users:
        info.add_user(u)
    
    domainId = 0
    for fname in rating_files:
        with open(fname, 'r') as f:
            lines = f.readlines()
        for line in lines:
            comp = line.split(',')
            uId = comp[0]
            if uId not in intersection_users:
                continue
            iId = comp[1]
            info.add_item(iId, domainId)
        domainId += 1
    
    return info

if __name__ == '__main__':
    folder = '../data/'
    rating_files = ['../data/ratings_Musical_Instruments.csv',
            '../data/ratings_Amazon_Instant_Video.csv']
    info = build_info(rating_files)
    
    info_path = '../data/user_item.pkl'
    with open(info_path, 'wb') as f:
        pickle.dump(info, f)
    print("Total users: " + str(info.getNumUsers()))
    print("num index:" + str(info.u_idx))

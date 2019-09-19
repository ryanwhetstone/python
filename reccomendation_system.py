import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data and format it
data = fetch_movielens(min_rating=4.0)

#print training and testing data
# print(repr(data['train']))
# print(repr(data['test']))

#create model
model = LightFM(loss='warp')
#train model
model.fit(data['train'], epochs=30, num_threads=1)

def sample_recommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape
    print(n_items)

    for user_id in user_ids:

        #movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        print(data['train'].tocsr()[user_id])

        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))

        top_items = data['item_labels'][np.argsort(-scores)]

        #print out the results
        print("User %s" % user_id)
        print("    Known positives:")

        for x in known_positives[:4]:
            print("          %s" % x)

        print("    Recommended:")

        for y in top_items[:4]:
            print("          %s" % y)

sample_recommendation(model, data, [3, 25, 141])
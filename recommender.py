import email
import numpy as np
import pandas as pd
import sys # can use sys to take command line arguments
from  tqdm import tqdm

from rec_functions import *

class Recommender():
    '''
    What is this class all about - write a really good doc string here
    '''
    def __init__(self, ):
        '''
        what do we need to start out our recommender system
        '''

    def fit(self, article_path,user_item_path,):
        '''
        fit the recommender to your dataset and also have this save the results
        to pull from when you need to make predictions
        '''
        self.interactions = pd.read_csv(user_item_path)
        self.articles = pd.read_csv(article_path)
        del self.interactions['Unnamed: 0']
        del self.articles['Unnamed: 0']

        #map the email in interactions to an id
        id_dict=email_mapper(self.interactions)
        self.interactions['user_id']=id_dict[0]
        self.email_id_map=id_dict[1]

        del self.interactions['email']


        self.user_item_matrix=create_user_item_matrix(self.interactions)


    def make_recs(self,email,rec_num,engine):
        '''
        given a user id or a movie that an individual likes
        make recommendations
        '''
        #convert the passed rec_num into int
        rec_num=int(rec_num)
        
        if email in self.email_id_map:
            print('the user is found in our database')
            user_id=self.email_id_map[email]

            if engine=='uu':#if user_user based collaborative filtering is used for recommendations
                neighours_df=get_top_sorted_users(user_id,self.interactions,self.user_item_matrix)

                df_group=self.interactions.groupby('article_id').count()['user_id'].reset_index().astype('int32')
                
                ids_seen=[int(float(i)) for i in get_user_articles(user_id,self.interactions,self.user_item_matrix)[0]]
                article_interactions=dict(zip(self.interactions['article_id'],self.interactions['user_id']))
                recs=[]
                
                for user in tqdm(neighours_df['neighbor_id']):
                    article_ids=set([int(float(i)) for i in get_user_articles(user,self.interactions,self.user_item_matrix)[0]]).difference(set(ids_seen))
                    articles_sorted=list(sorted(article_ids,key=lambda x:article_interactions[x],reverse=True))
                    recs.extend(articles_sorted)
                    
                rec_names=get_article_names(recs,self.interactions)
                
                return rec_names[:rec_num]
            
            if engine=='mf':#if matrix factorization based collaborative filtering is used for recommendations
                u, s, vt = np.linalg.svd(self.user_item_matrix)
                id_inter=np.around(np.dot(np.dot(u[self.user_item_matrix.index.to_numpy()==user_id,:50],np.diag(s[:50])),vt[:50,:]))

                rec_ids=self.user_item_matrix.columns.to_numpy()[id_inter.flatten()==1]
                if len(rec_ids)>=rec_num:
                    return get_article_names(rec_ids,self.interactions)[:rec_num]
                else:
                    return get_article_names(rec_ids,self.interactions)
        else:
            print('the user is not in our database, and we use the rank based recommendation engine')
            return get_top_articles(rec_num,self.interactions)                    

if __name__ == '__main__':
    # test different parts to make sure it works
    article_path='data/articles_community.csv'
    interactions_path='data/user-item-interactions.csv'
    
    #instantiate a recommender object and fit it to the dataset
    rec=Recommender()
    rec.fit(article_path,interactions_path)

    #make recommendations
    print(rec.make_recs(*sys.argv[1:]))


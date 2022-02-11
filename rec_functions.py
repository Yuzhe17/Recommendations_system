import pandas as pd
import numpy as np

def email_mapper(df):
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded,coded_dict

def get_top_articles(n, df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    interactions_by_title=df.groupby('title')['user_id'].count().sort_values(ascending=False)
    top_articles=list(interactions_by_title.index)[:n]
    
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    interactions_by_article=df.groupby('article_id')['user_id'].count().sort_values(ascending=False)
    top_articles=list(interactions_by_article.index)[:n]
    
    return top_articles # Return the top article ids

# create the user-article matrix with 1's and 0's

def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    # Fill in the function here
    df['has_interactions']=np.ones(df.shape[0])
    user_item=df.drop_duplicates().pivot(index='user_id',columns='article_id',
                                         values='has_interactions').fillna(0).astype(int)
    return user_item # return the user_item matrix 


def get_top_sorted_users(user_id, df, user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    
    user_interactions_df=df.groupby('user_id').count()['article_id'].reset_index()
    user_interactions=dict(zip(user_interactions_df['user_id'],user_interactions_df['article_id']))
    
    
    sim=np.dot(user_item.values,user_item.loc[user_id].values.reshape(-1,1)).flatten()
    
    neighbors_dict={'neighbor_id':list(user_item.index),'similarity':sim.flatten(),
               'num_interactions':[user_interactions[user_id] for user_id in user_item.index]}
    neighbors_df=pd.DataFrame(data=neighbors_dict).sort_values(['similarity','num_interactions'],ascending=False)
    return neighbors_df # Return the dataframe specified in the doc_string

def get_article_names(article_ids, df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    id_title=dict(zip(df['article_id'],df['title']))
    article_names=[id_title[float(article_id)] for article_id in article_ids]
    return article_names # Return the article names associated with list of article ids


def get_user_articles(user_id, df, user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    columns_array=np.array(user_item.columns,dtype=str)
    article_ids=columns_array[np.array(user_item.loc[user_id])==1]
    article_names=get_article_names(article_ids,df)
    return article_ids, article_names # return the ids and names




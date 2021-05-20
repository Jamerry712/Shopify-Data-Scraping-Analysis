#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json
from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np
import datetime


# In[2]:


# Testing website response == 200
url = 'https://undefeated.com/products.json'
response = requests.get(url)
if response.status_code == 200:
    print("Success")
else:
    print("Failure")


# In[12]:


# json.loads(response.content.decode('utf-8'))['products'][0]
# json.loads(response.content.decode('utf-8'))['products'][0]['variants']


# In[14]:


#Get products from a url
def getProducts(url):
    prod_dict = {'store_link':[],'product_id':[], 'product_title':[], 'handle':[],'product_link':[],'published_at':[],'vendor':[],'product_type':[],'sku_id':[],'sku_title':[],'sku':[],'price':[],'created_at':[],'updated_at':[]}
    response = requests.get(url+'products.json')
    # Check whether response fails
    if not response.status_code == 200:
        print('Response Error:',response.status_code)
    else:
        result_page = json.loads(response.content.decode('utf-8'))
        products = result_page['products']
        for x in range(len(products)):
            prod_id = products[x]['id']
            prod_title = products[x]['title']
            handle = products[x]['handle']
            link = (url + 'products/{}'.format(handle))
            pub_at = products[x]['published_at']
            vendor = products[x]['vendor']
            prod_type = products[x]['product_type']
            variants = products[x]['variants']
            for variant in variants:
                prod_dict['store_link'].append(url)
                prod_dict['product_id'].append(prod_id)
                prod_dict['product_title'].append(prod_title)
                prod_dict['handle'].append(handle)
                prod_dict['product_link'].append(link)
                prod_dict['published_at'].append(pub_at)
                prod_dict['vendor'].append(vendor)
                prod_dict['product_type'].append(prod_type)
                prod_dict['sku_id'].append(variant['id'])
                prod_dict['sku_title'].append(variant['title'])
                prod_dict['sku'].append(variant['sku'])
                prod_dict['price'].append(variant['price'])
                prod_dict['created_at'].append(variant['created_at'])
                prod_dict['updated_at'].append(variant['updated_at'])
            
        df = pd.DataFrame.from_dict(prod_dict)
        return df

#Get products from all urls
def get_all_products(urls):
    df_all = getProducts(urls[0])
    for url in urls[1:]:
        df = getProducts(url)
        df_all = pd.concat([df_all, df])
    return df_all


# In[4]:


#Read url_list
urls = []
shopify_links = open('shopify_links.txt', 'r')
for slinks in shopify_links:
    slinks = slinks.strip('\n')
    urls.append(slinks)


# In[5]:


start_time = datetime.datetime.now()
shopify = get_all_products(urls)
end_time = datetime.datetime.now()
print(end_time - start_time)


# In[6]:


shopify.to_csv('shopify_products.csv')
shopify


# In[11]:


shopify.info()


# In[7]:


word_lists = shopify['product_title'].apply(lambda x: x.split(" "))


# In[10]:


from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
extended_stop = ['For','for','And','0',"'d", "'ll", "'re", "'s", "'ve", 'could', 'might', 'must',                   "n't", 'need', 'sha', 'wo', 'would','llc','inc','not','co','ltd','com','corp','m',                  '','be','ig','have','£','tweet','st','pm','let','\u2066\u2069','iam','w',                  'ma','t','v','eth','c','','b','d','bc']
stop = stopwords.words("english") + list(string.punctuation) + extended_stop

words_dict = dict()
for l in word_lists:
    for w in l:
        if w not in extended_stop:
            if words_dict.get(w):
                words_dict[w] += 1
            else:
                words_dict[w] = 1

results = sorted(words_dict.items(),key=lambda x: x[1],reverse=True) 

from wordcloud import WordCloud

wc_dict = words_dict.copy()
# wc_dropped = ["invest","financial","finance",'planning']
# for w in wc_dropped:
#     wc_dict.pop(w)
    
wc = WordCloud(background_color='white',width=2000,height=1000)
wc.generate_from_frequencies(wc_dict)
plt.imshow(wc)
plt.axis("off")
plt.show()

# wc.to_file('wc.png')


# In[ ]:





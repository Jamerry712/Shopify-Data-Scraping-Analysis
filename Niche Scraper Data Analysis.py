#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy                 as np
import pandas                as pd
import matplotlib.pyplot     as plt
import datetime
import seaborn               as sns


# In[2]:


#read the Niche Scraper data
BH = pd.read_csv('Niche Scraper - Beauty _ Health.csv', encoding='latin1')


# In[3]:


BH.info()


# # WordClouds
# ## This could help verify whether the data belongs to the right category

# In[4]:


word_lists = BH['product_name'].apply(lambda x: x.split(" "))


# In[5]:


from nltk.corpus import stopwords
import string
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

wc.to_file('wc-BH.png')


# # Data Cleaning

# In[6]:


BH.head(5)


# In[7]:


#clean the text
def clean_text(text):
    drop_list=['\n','+','  ','Total Orders','Recent Orders','The number of new orders this week']
    for i in drop_list:
        text = text.replace(i,'')
    text = text.strip(' ')
    return text

#Turn orders into integer
def int_orders(order):
    order = int(order.replace(',',''))
    return order

#Turn price range into average price
def avg_price(price):
    if '-' in price.split(' '):
        lower_p_str = price.split(' ')[0]
        upper_p_str = price.split(' ')[2]
        lower_p_num = float(lower_p_str.split('$')[1])
        upper_p_num = float(upper_p_str.split('$')[1])
        avg_price = (lower_p_num + upper_p_num) / 2
    else:
        avg_price = float(price.split(' ')[0].split('$')[1])
    return avg_price

#data cleaning
def clean_data(df):
    # clean the string variables
    for index in df.columns[:-4]:
        df[index] = df[index].apply(clean_text)
    
    # turn orders into int variables
    df['total_orders'] = df['total_orders'].apply(int_orders)
    df['recent_orders'] = df['recent_orders'].apply(int_orders)

    # calculate the average price
    df['avg_price'] = df['price'].apply(avg_price)
    
    #calculate the GMVs
    df['total_GMV'] = df['avg_price']*df['total_orders']
    df['recent_GMV'] = df['avg_price']*df['recent_orders']
    return df


# In[8]:


BH = clean_data(BH)
BH


# # Visualization

# ## Distribution of Countries

# In[10]:


from matplotlib import font_manager as fm
from  matplotlib import cm

def dist_country(df, category):
    country_l = df['country'].unique()
    country_count = np.array([])
    for country in country_l:
        country_count = np.append(country_count, len(df[df['country']==country]))

    country_a = np.array(country_l)
    plt.figure(figsize=(10,6))
    plt.bar(country_a[np.argsort(country_count)][-15:], country_count[np.argsort(country_count)][-15:],fc='lightcoral')
    for a,b in zip(country_a[np.argsort(country_count)][-15:], country_count[np.argsort(country_count)][-15:]):
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=12)
    plt.title('Top15 Countries of %s products'%category, fontsize=16)
    plt.xticks(country_a[np.argsort(country_count)][-15:],size='large',rotation=0)
    plt.xlabel('Country',fontsize=14)
    plt.ylabel('Count',fontsize=14)
    plt.show()
    

def pie_country(df):
    country_l = df['country'].unique()
    country_count = np.array([])
    for country in country_l:
        country_count = np.append(country_count, len(df[df['country']==country]))

    country_a = np.array(country_l)
    
    #to draw the pie chart
    others = np.sum(country_count) - np.sum(country_count[np.argsort(country_count)][-15:])
    labels = np.insert(country_a[np.argsort(country_count)][-15:], 0, ['Others'])
    sizes = np.insert(country_count[np.argsort(country_count)][-15:], 0, others)
    explode = (0, 0,0,0,0,0,0,0,0,0,0,0,0.1,0,0,0) 
    # "explode" ， 非0的值显示突出的slice，值越大，离中心越远
    fig, axes = plt.subplots(figsize=(16,10),ncols=2) # 设置绘图区域大小
    ax1, ax2 = axes.ravel()

    colors = cm.rainbow(np.arange(len(sizes))/len(sizes)) # colormaps: Paired, autumn, rainbow, gray,spring,Darks
    patches, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.0f%%',explode=explode,
            shadow=False, startangle=170, colors=colors, labeldistance=1.2,pctdistance=0.7, radius=0.4)
    # labeldistance: 控制labels显示的位置
    # pctdistance: 控制百分比显示的位置
    # radius: 控制切片突出的距离

    ax1.axis('equal')  

    # 重新设置字体大小
    proptease = fm.FontProperties()
    proptease.set_size('xx-large')
    # font size include: ‘xx-small’,x-small’,'small’,'medium’,‘large’,‘x-large’,‘xx-large’ or number, e.g. '12'
    plt.setp(autotexts, fontproperties=proptease)
    plt.setp(texts, fontproperties=proptease)

    ax1.set_title('Country Percentage Pieplot', loc='center',fontsize=30)

    # ax2 只显示图例（legend）
    ax2.axis('off')
    ax2.legend(patches, labels, loc='center left',prop={'size': 16})

    plt.tight_layout()
    # plt.savefig("pie_shape_ufo.png", bbox_inches='tight')
    plt.savefig('Country_Dist_Pie.jpg')
    plt.show()


# In[11]:


dist_country(BH,'Beauty & Health')


# In[12]:


pie_country(BH)


# ### Average Price Distribution

# In[13]:


def dist_price(df,category):
    plt.figure(figsize=(20,5))
    plt.hist(df['avg_price'],bins=500, density=False)
    plt.title('Overall Distribution of Average Price of %s'%category,fontsize=20)
    plt.xlabel('Average Price',fontsize=16)
    plt.ylabel('Count',fontsize=16)
    plt.show() 
    num_total = len(df)
    num_50 =len(df[df['avg_price']<=50])
    num_100 =len(df[df['avg_price']<=100])
    print('The number of %s products <=50 USD is'%category, num_50, ', making up',  '{0:.2%}'.format(num_50/num_total), 'of total')
    print('The number of %s products <=100 USD is'%category, num_100, ', making up',  '{0:.2%}'.format(num_100/num_total), 'of total')
    print('The max price for %s products is:'%category, np.max(df['avg_price']))
    print('The most expensive product is',df['product_name'].iloc[np.where(df['avg_price']==np.max(df['avg_price']))[0][0]])


# In[14]:


dist_price(BH, 'Beauty & Health')


# In[15]:


def dist_price_div(df,category):
    plt.figure(figsize=(20,10))
    plt.subplot(221)
    plt.hist(df[df['avg_price']<=1]['avg_price'],bins=100)
    plt.title('Distribution of Average Price(0-1 USD) of %s'%category,fontsize=15)
    plt.xlabel('Average Price')
    plt.ylabel('Count')

    plt.subplot(222)
    plt.hist(df[(df['avg_price']>1)&(df['avg_price']<=50)]['avg_price'],bins=100)
    plt.title('Distribution of Average Price(1-50 USD) of %s'%category,fontsize=15)
    plt.xlabel('Average Price')
    plt.ylabel('Count')

    plt.subplot(223)
    plt.hist(df[(df['avg_price']>50)&(df['avg_price']<=100)]['avg_price'],bins=50)
    plt.title('Distribution of Average Price(50-100 USD) of %s'%category,fontsize=15)
    plt.xlabel('Average Price')
    plt.ylabel('Count')

    plt.subplot(224)
    plt.hist(df[(df['avg_price']>100)&(df['avg_price']<500)]['avg_price'],bins=50)
    plt.title('Distribution of Average Price(100-500 USD) of %s'%category,fontsize=15)
    plt.xlabel('Average Price')
    plt.ylabel('Count')

    plt.show()


# In[16]:


dist_price_div(BH,'BH')


# In[17]:


print('The percentage of products <=50 USD is','{0:.2%}'.format(len(BH[(BH['avg_price']<=50)])/len(BH)))


# ### Total Orders Distribution

# In[18]:


def dist_orders(df, order='total_orders'):
    plt.figure(figsize=(20,5))
    plt.hist(df[df[order]>0][order],bins=100,fc='green',density=False)
    plt.title('Overall Distribution of %s(>0)'%order,fontsize=20)
    plt.xlabel('%s'%order, fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.show()
    
    print('The max %s is'%order, max(df[order]))
    print('The best seller is:',df['product_name'].iloc[np.where(df[order]==np.max(df[order]))[0][0]])
    print('The 90th percentage is', np.percentile(df[order], 90))
    print('The 95th percentage is', np.percentile(df[order], 95))


# In[19]:


dist_orders(BH,'recent_orders')


# In[20]:


def dist_orders_div(df, order='total_orders'):    
    per_90 = np.percentile(df[order], 90)
    per_95 = np.percentile(df[order], 95)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.hist(df[(df[order]>0)&(df[order]<=per_90)][order],bins=50,fc='green')
    plt.title('Distribution of %s(last 90 percent)'%order,fontsize=15)
    plt.xlabel('%s'%order)
    plt.ylabel('Count')

    plt.subplot(132)
    plt.hist(df[(df[order]>per_90)&(df[order]<=per_95)][order],bins=50,fc='green')
    plt.title('Distribution of %s(top 5-10 percent)'%order,fontsize=15)
    plt.xlabel('%s'%order)
    plt.ylabel('Count')

    plt.subplot(133)
    plt.hist(df[df[order]>per_95][order],bins=50,fc='green')
    plt.title('Distribution of %s(top 5 percent)'%order,fontsize=15)
    plt.xlabel('%s'%order)
    plt.ylabel('Count')

    plt.show()


# In[21]:


dist_orders_div(BH, order='recent_orders')


# ### Aliscore Distribution

# In[26]:


def dist_aliscore(df):
    plt.figure(figsize=(20,5))
#     plt.figure(figsize=(20,5))
    plt.hist(df['aliscore'],bins=100,fc='darkorange')
    plt.title('Overall Distribution of Aliscore',fontsize=20)
    plt.axvline(np.percentile(df['aliscore'], 50), c='r')
    plt.axvline(np.percentile(df['aliscore'], 90), c='r')
    plt.xlabel('Aliscore',fontsize=15)
    plt.ylabel('Count',fontsize=15)
    plt.show()
    print('The max Aliscore is', max(df['aliscore']))
    print('The 90th percentage is', np.percentile(df['aliscore'], 90))
    print('The 95th percentage is', np.percentile(df['aliscore'], 95))

def dist_aliscore_40(df):    
    plt.figure(figsize=(20,5))
    plt.hist(df[df['aliscore']>=40]['aliscore'],bins=100,fc='darkorange')
    plt.title('Distribution of Aliscore >=40',fontsize=20)
    plt.xlabel('Aliscore',fontsize=15)
    plt.ylabel('Count',fontsize=15)
    plt.show()


# In[27]:


dist_aliscore(BH)


# In[29]:


dist_aliscore_40(BH)


# #### Correlation of Aliscore

# In[30]:


BH.corr()['aliscore'].sort_values().plot(kind='barh',color='darkorange')
plt.title('Correlation map')
plt.show()


# ### Competition Score Distribution 

# In[32]:


def dist_com(df):
    plt.figure(figsize=(20,5))
    plt.hist(df['competition'],bins=80,fc='crimson')
    plt.title('Overall Distribution of Competition Score',fontsize=20)
    #you can manually set this threshold to filter out abnormal data
    plt.axvline(29.5, c='b')
    plt.xlabel('Competition Score',fontsize=15)
    plt.ylabel('Count',fontsize=15)
    plt.show()
dist_com(BH)


# In[36]:


print('The num of data <threshold is',len(BH[BH['competition']<=29.5]))
print('making up','{0:.2%}'.format(len(BH[BH['competition']<=34])/len(BH)))


# #### Correlation of Competition Score

# In[38]:


BH.corr()['competition'].sort_values().plot(kind='barh',color='crimson')
plt.title('Correlation map')
plt.show()


# In[38]:


plt.figure(figsize=(20,5))
plt.hist(BH['total_GMV'],bins=100,fc='cadetblue')
plt.title('Overall Distribution of recent_GMV',fontsize=20)
# plt.axvline(34.35, c='r')
# plt.axvline(52.68, c='r')
plt.xlabel('Total GMV')
plt.ylabel('Count')
plt.show()


# In[39]:


print(np.mean(BH['total_GMV']))
print(np.max(BH['total_GMV']))
print(np.percentile(BH['total_GMV'],50))
print(np.percentile(BH['total_GMV'],90))
print(np.percentile(BH['total_GMV'],95))


# # Set the right threshold for target list

# Let's assume our target group for Customer Electronics has the Competition <= 30, shipping destination: US
# 
# Sort the target list by Competition

# In[39]:


target_BH=BH[(BH['country'] == 'US')&(BH['competition']<=30)].sort_values('competition',ascending=True).drop(columns=['recentorders','sellwithtopdser','todser_link'])
target_BH.to_csv('targetlist_Beauty & Health.csv')
target_BH


# target data summary

# In[40]:


target_BH.describe()


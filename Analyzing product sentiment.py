
# coding: utf-8

# In[122]:

import graphlab


# #Read product review data

# In[123]:

products = graphlab.SFrame('amazon_baby.gl/') 


# #lets explore this data together

# In[124]:

products.head()


# #Build the word count vector for each review

# In[125]:

products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[126]:

products.head()


# In[127]:

graphlab.canvas.set_target('ipynb')


# In[128]:

products['name'].show()


# #Explore Vulli Sophie

# In[129]:

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[130]:

len(giraffe_reviews)


# In[131]:

giraffe_reviews['rating'].show('Categorical')


# #Build a sentiment classifier

# In[132]:

products['rating'].show(view='Categorical')


# #Define what is a positive and a negative sentiment

# In[133]:

#ignore all 3* reviews
products = products[products['rating'] != 3]


# In[134]:

#positive sentiment = 4* or 5* reviews Here we are creating our binary y result (good or bad review)
products['sentiment'] = products['rating'] >= 4


# In[135]:

products.head()


# #lets train the sentiment classifier

# In[136]:

train_data,test_data = products.random_split(.8, seed=0)


# In[170]:

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)


# #Evaluate the sentiment model

# In[138]:

sentiment_model.evaluate(test_data, metric='roc_curve')


# In[139]:

sentiment_model.show(view='Evaluation')


# #apply the learned model to understand sentiment for our friend the Giraffe

# In[140]:

giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type = 'probability')


# In[141]:

giraffe_reviews.head()


# #sort the reviews based on predicted sentiment and explore

# In[142]:

giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[143]:

giraffe_reviews.head()


# In[144]:

giraffe_reviews[0]['review']


# In[145]:

giraffe_reviews[1]['review']


# #Show most negative reviews

# In[146]:

giraffe_reviews[-1]['review']


# In[147]:

giraffe_reviews[-2]['review']


# #1 Often, ML practitioners will throw out words they consider “unimportant” before training their model. This procedure can often be helpful in terms of accuracy. Here, we are going to throw out all words except for the very few above. Using so few words in our model will hurt our accuracy, but help us interpret what our classifier is doing.

# In[148]:

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']


# In[151]:

#define all functions to fetch word_counts
def awesome_count (dict1):
    if 'awesome' in dict1:
        return dict1['awesome']
    else:
        return 0

def great_count (dict1):
    if 'great' in dict1:
        return dict1['great']
    else:
        return 0
    
def fantastic_count (dict1):
    if 'fantastic' in dict1:
        return dict1['fantastic']
    else:
        return 0
    
def amazing_count (dict1):
    if 'amazing' in dict1:
        return dict1['amazing']
    else:
        return 0

def love_count (dict1):
    if 'love' in dict1:
        return dict1['love']
    else:
        return 0

def horrible_count (dict1):
    if 'horrible' in dict1:
        return dict1['horrible']
    else:
        return 0
    
def bad_count (dict1):
    if 'bad' in dict1:
        return dict1['bad']
    else:
        return 0
    
def terrible_count (dict1):
    if 'terrible' in dict1:
        return dict1['terrible']
    else:
        return 0
    
def awful_count (dict1):
    if 'awful' in dict1:
        return dict1['awful']
    else:
        return 0

def wow_count (dict1):
    if 'wow' in dict1:
        return dict1['wow']
    else:
        return 0
    
def hate_count (dict1):
    if 'hate' in dict1:
        return dict1['hate']
    else:
        return 0
    


# In[152]:

#Create column counts for selected words
products['awesome'] = products['word_count'].apply(awesome_count)
products['great'] = products['word_count'].apply(great_count)
products['fantastic'] = products['word_count'].apply(fantastic_count)
products['amazing'] = products['word_count'].apply(amazing_count)
products['love'] = products['word_count'].apply(love_count)
products['horrible'] = products['word_count'].apply(horrible_count)
products['bad'] = products['word_count'].apply(bad_count)
products['terrible'] = products['word_count'].apply(terrible_count)
products['awful'] = products['word_count'].apply(awful_count)
products['wow'] = products['word_count'].apply(wow_count)
products['hate'] = products['word_count'].apply(hate_count)


# In[116]:

awesome_products = products[products['awesome'] == 2]


# In[117]:

awesome_products[1]['awesome']


# In[157]:

print("awesome: " + str(products['awesome'].sum()))
print("great: " + str(products['great'].sum()))
print("fantastic: " + str(products['fantastic'].sum()))
print("love: " + str(products['love'].sum()))
print("horrible: " + str(products['horrible'].sum()))
print("bad: " + str(products['bad'].sum()))
print("terrible: " + str(products['terrible'].sum()))
print("awful: " + str(products['awful'].sum()))
print("wow: " + str(products['wow'].sum()))
print("hate: " + str(products['hate'].sum()))


# In[158]:

#ANSWER 1
#awesome: 2002
#great: 42420
#fantastic: 873
#love: 40277
#horrible: 659
#bad: 3197
#terrible: 673
#awful: 345
#wow: 131
#hate: 1057


# In[161]:

train_data,test_data = products.random_split(.8, seed=0)


# In[164]:

selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features= selected_words,
                                                     validation_set=test_data)


# In[165]:

selected_words_model['coefficients']


# In[168]:

selected_words_model['coefficients'].sort('value', ascending=False, rows = 12 )


# #Most Positive: love
# #Most Negative: awful

# In[169]:

selected_words_model.evaluate(test_data)


# #Selected_Words_Model accuracy: 0.8431419649291376
# #Sentiment_model Accuracy:  0.916256305548883
# #Majority Class = Positive
# #What is the baseline approach? 50/50 since this is a binary classification. If we predict the majority class @ 100% then we have an error of 15.88% which is more than our new selected_words_model with an error of 15.6%

# In[171]:

sentiment_model.evaluate(test_data)


# In[173]:

products['sentiment'].show('Categorical')


# In[176]:

diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']


# In[178]:

diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type = 'probability')


# In[179]:

diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment', ascending=False)


# In[180]:

diaper_champ_reviews.head()


# #predicted sentiment for most positive review according to sentiment model: 
# 0.999999937267
# 0.7969408512906704
# 

# In[181]:

selected_words_pred = selected_words_model.predict(diaper_champ_reviews[0:1], output_type='probability')


# In[182]:

selected_words_pred


# In[185]:

diaper_champ_reviews[1]


# In[ ]:

#only 1 count of love in the selected words model but this is a long review 


# In[187]:

selected_words_model['coefficients'].print_rows(num_rows = 12)


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# In[21]:


from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')


# In[2]:


file=open('nlp.txt','r')
text=file.read()
print(text)


# In[3]:


sentences=sent_tokenize(text)


# In[4]:


print("number of sentences:",len(sentences))
for i in  range(len(sentences)):
    print("\nSentence",i+1,":\n",sentences[i])


# In[5]:


from nltk.tokenize import word_tokenize


# In[6]:


words=word_tokenize(text)


# In[7]:


print("total number of words:",len(words))
print(words)


# In[8]:


words=word_tokenize(text,preserve_line=True)
len(words)


# In[9]:


from nltk.tokenize import word_tokenize


# In[10]:


file=open('nlp.txt','r')
text=file.read()


# In[11]:


words=word_tokenize(text)
len(words)


# In[12]:


from nltk.probability import FreqDist
all_fdist=FreqDist(words).most_common(20)
print(all_fdist)


# In[13]:


import matplotlib.pyplot as plt
import pandas as pd
all_fdist=pd.Series(dict(all_fdist))
fig,ax=plt.subplots(figsize=(5,5))
all_fdist.plot(kind='bar')
plt.title('Frequency Distribution of words')
plt.ylabel('Count')
plt.savefig('a.jpg')


# In[14]:


text=text.lower()


# In[15]:


import re
text=re.sub('[^A-Za-z0-9]+',' ',text)


# In[16]:


text=re.sub('\S*\d\S*','',text).strip()


# In[17]:


from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
stopwords=set(STOPWORDS)
wordcloud=WordCloud(width=800,height=800, background_color='white', stopwords=stopwords,min_font_size=10).generate(text)
plt.figure(figsize=(5,5),facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[26]:


from skimage.io import imread
cloud=imread('Cobra.jpg')
plt.imshow(cloud)


# In[23]:


from skimage.io import imread
tree=imread('Tree.jpg')
plt.imshow(tree)


# In[22]:


from skimage.io import imread
cloud1=imread('Cloud1.jpg')
plt.imshow(cloud1)


# In[27]:


from skimage.io import imread
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Load the image for the mask
try:
    cloud = imread('cobra.jpg')
except FileNotFoundError:
    print("File 'cloud.png' not found. Please check the path.")
    cloud = None  # Set cloud to None if the file is not found

if cloud is not None:
    # Define stopwords
    stopwords = set(STOPWORDS)

    # Create the word cloud
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color='white',
        stopwords=stopwords,
        min_font_size=10,
        mask=cloud
    ).generate(text)

    # Display the word cloud
    plt.figure(figsize=(5,5), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

    # Save the word cloud to a file
    wordcloud.to_file('abc.png')
else:
    print("Cannot generate word cloud as 'cloud.png' is not defined.")


# In[28]:


from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
stopwords=set(STOPWORDS)
wordcloud=WordCloud(width=800,height=800,
background_color='white',
stopwords=stopwords,min_font_size=10,mask=cloud1).generate(text)
plt.figure(figsize=(5,5),facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[29]:


from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
stopwords=set(STOPWORDS)
wordcloud=WordCloud(width=800,height=800, background_color='white', stopwords=stopwords,min_font_size=10,mask=tree).generate(text)
plt.figure(figsize=(5,5),facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# # Spell Correction

# In[30]:


import nltk
from nltk.metrics.distance import edit_distance


# In[31]:


nltk.download('words')
from nltk.corpus import words
cw=words.words()


# In[32]:


iw=['happpy','amzzzzzing','intelliegent']
for word in iw:
    temp=[(edit_distance(word,w),w)for w in cw if w[0]==word[0]]
print(sorted(temp,key=lambda val:val[0])[0][1])


# # Stemming

# In[33]:


from nltk.tokenize import word_tokenize
file=open('nlp.txt','r')
text=file.read()
text=text.lower()
import re
text=re.sub('[^A-Za-z0-9]',' ',text)
text=re.sub('\s*\d\s*','',text).strip()
print(text)


# In[34]:


words=word_tokenize(text,preserve_line=True)
print(words)


# In[35]:


from nltk.stem import PorterStemmer
ps=PorterStemmer()
ps_stem_sent=[ps.stem(words_sent)for words_sent in words]
print(ps_stem_sent)


# # Lemmatization

# In[36]:


print(words)


# In[38]:


import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
l=WordNetLemmatizer()
ls=[l.lemmatize(words_sent)for words_sent in words]
print(ls)


# In[39]:


from nltk.stem import WordNetLemmatizer
l=WordNetLemmatizer()
print('rocks:',l.lemmatize('rocks'))
print('corpora:',l.lemmatize('corpora'))
print('better:',l.lemmatize('better',pos='a'))


# #  Parts of speech tagging

# In[42]:


import nltk
from nltk import word_tokenize
import nltk
nltk.download('averaged_perceptron_tagger')


# In[44]:


text='I am very hungry but stomak is empty'
words=word_tokenize(text)
print('parts of speech:',nltk.pos_tag(words))


# # Vectorization

# In[45]:


from sklearn.feature_extraction.text import CountVectorizer
s=['He is smart boy.she is also smart', 'chirag and man is a smart persons']


# In[46]:


cv=CountVectorizer()
x=cv.fit_transform(s)
x=x.toarray()
v=sorted(cv.vocabulary_.keys())
print(v)
print(x)


# In[47]:


cv=CountVectorizer(ngram_range=(2,2))
x=cv.fit_transform(s)
x=x.toarray()
v=sorted(cv.vocabulary_.keys())
print(v)
print(x)


# In[48]:


from sklearn.feature_extraction.text import TfidfVectorizer
s=['corona virus is a highly infectious disease',
'corona virus affects older people the most',
'older people are at high risk due to this disease']


# In[49]:


tfidf=TfidfVectorizer()
t=tfidf.fit_transform(s)
import pandas as pd
df=pd.DataFrame(t[0].T.todense(),
index=tfidf.get_feature_names_out(),
columns=['TF-IDF'])
df=df.sort_values('TF-IDF',ascending=False)
df


# In[50]:


import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
text="""Natural language processing refers to the branch of 
        computer science concerned with giving computers
        the ability to understand text"""
words=word_tokenize(text)
output=list(ngrams(words,2))
output


# In[51]:


import matplotlib.pyplot as plt
x=[1,1,2,3,3,4,5,6,7,7,8,8,9,10,10,9,9,11,11,11,12,12]
y=[1,3,2,3,1,1,3,1,1,3,1,3,3,3,1,1,2,1,3,1,1,3]
plt.plot(x,y,'*--b')
plt.show()


# In[ ]:





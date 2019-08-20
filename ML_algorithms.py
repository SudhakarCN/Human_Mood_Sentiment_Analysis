import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_extraction.text import CountVectorizer
import pickle

#Data Reading:
df = pd.read_csv("C:\\Users\\sudhakarceemavaram_n\\Desktop\\Dataset\\Kaggle\\train_data.csv")
stop = stopwords.words("english")

#Data Cleaning:
def cleaning(raw):
    alpha_only = re.sub('[^a-zA-Z]'," ",raw)
    all_lower = alpha_only.lower()
    words = all_lower.split()
    for i in words:
        if i in stop:
            words.remove(i)
    return " ".join(words)

x = list(map(cleaning,df.content))
y = df["sentiment"]
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1,random_state=7)


#Vectorization:
vectx = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(xtrain)
vecty = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(ytrain)

xtrainvector = vectx.transform(xtrain)
ytrainvector = vecty.transform(ytrain)

#print(type(xtrainvector))
#print(type(ytrainvector))

#Model:
model = LogisticRegression(penalty="l2",solver="lbfgs",multi_class="multinomial")
model.fit(xtrainvector,ytrain)

predict = model.predict(vectx.transform(xtest))

score = accuracy_score(predict,ytest)
report = classification_report(ytest,predict)

print("The predicted value is: ",predict)
print("The accuracy score is: ",score)
print("Report: ",report)

#Saving the pickle file:
#file = "Human_mood.model"
#pickle.dump(model,open(file,"wb"))

'''
df1 = pd.read_csv("C:\\Users\\sudhakarceemavaram_n\\Desktop\\Dataset\\Kaggle\\test_data.csv")
x1 = list(map(cleaning,df1.content))
xvector = vectx.transform(x1)
loadmodel = pickle.load(open("C:\\Users\\sudhakarceemavaram_n\\PycharmProjects\\pygame\\object\\Human_mood.model","rb"))
value = list(loadmodel.predict(xvector))
df1["Sentiment_updated"] = value
print(df1.head())

df1.to_csv("C:\\Users\\sudhakarceemavaram_n\\Desktop\\Dataset\\Kaggle\\Updated_Sentiments.csv")
'''

import pandas as pd
import re
from nltk.corpus import stopwords
from textblob import TextBlob
stop = stopwords.words('english')


str='''calculation guidelines calculate thd nearer calculating percentage
detailscomments outage advice accompanying identifications associated raceway
keys electrical misc tekapeek preferably code besides going allows
recommend veronica availablility referance anodizing propeller marketing
systems business calgary anozided anderson replication continuation
 shriharsh despande kicked housings jiten vadher psl lip wi ams  pso cwi market
boreback defined thick karuna temperatures taps clam austenite kind
requiredsulfur sulfur tdr heate window packaging boxes satisfied deviate
shipments acrylic urethane health pirtdrbom  clippedblunt  gmcc dd jul krylova
ryerson castle metals ka  gagan sidhu under  drawingtdr'''

list=[]
for x in str.split():
    list.append(x)


df=pd.Series(list)

df=df.str.upper()

def check_func(data):
    df=df.apply(lambda x: TextBlob(x).correct()) #For checking grammer 



print(df)




#df=pd.DataFrame(list, columns=['COL1'])




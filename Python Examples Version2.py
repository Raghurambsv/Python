'''
dir(list) or dir(<any listname>) gives the list of all the methods you can apply on them

dir(str) or dir(<any string name>) gives you list of all methods you can apply on string
'''

## Date and timestamp related
from datetime import datetime
delta = datetime.now() - datetime(2017,7,9)
print(delta.days,delta.seconds)

now=datetime.now()
then=datetime(2016,9,23)
time=now - then

print('The time now is ',time)

#formatting of  date
whenever= datetime.strptime('2017-12-24','%Y-%m-%d')
print(whenever)


whenever=datetime.now()
print(whenever.strftime('%Y'))



## ITERATE MULTIPLE LISTS or SEQUENCES

a = ['a','b','c','d']
b = [1,2,3,4]
c = ['aa','bb','cc','dd']
for i,j,k in zip(a,b,c):
    print('%s is %s and %s'% (i,j,k))
    

## concatenating file contents

import glob2
from datetime import datetime 
filename=glob2.glob("//Users//raghuram.b//Desktop//Python//file*")

Newfile=str(datetime.now()) + ".txt"

with open("//Users//raghuram.b//Desktop//Python//"+Newfile,'w') as Newfile:
    for file in filename:
        with open (file,'r') as f:
            Newfile.write(f.read() + "\n")

print("job done")            
                



    
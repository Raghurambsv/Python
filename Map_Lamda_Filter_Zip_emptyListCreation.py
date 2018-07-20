#to find out which version of python you running
import sys
print("The version of python is",sys.version)

####Flaten a list
l = [[1, 2, 3], [4, 5, 6], [7], [8, 9]]
flat_list = [item for sublist in l for item in sublist]
print(flat_list)
    
##Create empty list of 10 values (you can intialize the values to None or anything you like)    
x=10
a=[None]*x
print(a)
#or
b=[i for i in range(x)]
print(b)
c=[None for i in range(x)]
print(c)


#################MAP###################
#map(fun, iter)
def addition(n):
    return n + n
# We double all numbers using map()
numbers = (1, 2, 3, 4)
result = map(addition, numbers)
for ch in result:
    print(ch)
#################lambda################

# lambda functions with NO_ARGS 
var=lambda :print("Hello")
print(var())
# lambda functions with ONE ARGS 
downbytwo=lambda e:e-2
print(downbytwo(5))

# lambda functions with TWO ARGS ( Note: You can pass "N" no of ARGS but but only one expression you can give) 
add=lambda x,y:x+y
print (add(5,6)) 

# lambda function with DEFAULT ARGS
var=lambda x=1,y=2: x+y
print(var())

      
##lambda with map & list (with Iterables example )
for ch in map(lambda x:x*2,[10,20,30,40]):
    print(ch)


##lambda with map & dict
dict_a=[{'name':'python','points':10},{'name':'java','points':8}]
for ch in map(lambda x:x['name'],dict_a):
    print(ch)
for ch in map(lambda x:x['points']*200  ,dict_a):
    print(ch)
   
for ch in map(lambda x:x['name']=='python',dict_a):
    print(ch)    
    
    
    
    
##multiple iterables 
list_a=[1,2,3,4,5]    
list_b=[10,20,30,40,50]
list_c=[100,200,300] ## all iterables should be of same length if not automatically it gives output only till least LIST LENGTH
for ch in map(lambda x,y,z:x+y+z ,list_a,list_b,list_c):
    print(ch)
#Neither we can access the elements of the map object with index nor we can use len() to find the length of the map object
#we can do it by running thru forloop & (other methods need to find out)    

    
############FILTER############### #######(No multiple iterables in filter)
#filter function expects two arguments, function_object and an iterable. 
#function_object returns a boolean value.
# function_object is called for each element of the iterable and filter returns only those element for which the function_object returns true.
#Like map function, filter function also returns a list of element. Unlike map function filter function can only have one iterable as input.
    
#filter(FUNCTION,ITERABLE)    
a = [1, 2, 3, 4, 5, 6,62,63,64,102,113,116]
list2=filter(lambda x:x % 2 == 0,a)##checking for even number only
for ch in list2:
    print(ch) ##prints only even numbers with help of filter
#or   (list comprehension with condition) 
a = [1, 2, 3, 4, 5, 6,62,63,64,102,113,116]
result=[item for item in a if item % 2 == 0]
print(result) 

#or (By creating function) in this case lambda is better
a = [1, 2, 3, 4, 5, 6,62,63,64,102,113,116]
def even_num(x):
    if x % 2 == 0:
        return x
    else:
        return False

result=filter(even_num,a) 
for ch in result:
    print(ch)   


        
###############Zip function############### 
list_a = [1, 2, 3, 4, 5]
list_b = ['a', 'b', 'c', 'd', 'e']
result=zip(list_a,list_b)
for x,y in result:
    print(x,y)#print('%d%s'%(x,y))
    
result=zip(list_a,list_b)
for item in result:  
    print(item)
    
##Alternate for zip -manually creation  
list_a = [1, 2, 3, 4, 5]
list_b = ['a', 'b', 'c', 'd', 'e']
retList=[]
for i in range(5):
    retList.append((list_a[i],list_b[i])) 
    
print(retList)    
        
    
#############Reduce####################
#The reduce() function takes two parameters- a function, and a list. 
#It performs computation on sequential pairs in the list and returns one output. 
#But to use it, you must import it from the functools module.   
numbers=[0,1,2,3,4,5,6,7,8,9,10]
from functools import reduce
print(reduce(lambda x,y:y-x,numbers)) 
#1-0=0
#
#2-1=1
#
#3-1=2
#
#4-2=2
#
#5-2=3
#
#6-3=3
#
#7-3=4
#
#8-4=4
#
#9-4=5
#
#10-5=5
#
#Hence, it outputs 5.   



 
 
 
 
 



    
   
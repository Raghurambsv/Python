########################################Iterator & GENERATORS###############################################   
simple_list = [1, 2, 3]
my_iterator = iter(simple_list) #iter method is called on the object to converts it to an iterator object.
print(my_iterator)
print(next(my_iterator)) #the next() method is called on the iterator object to get the next element of the sequence.
print(next(my_iterator))  
#A StopIteration exception is raised when there are no elements left to call.


#Generator
#Generator is an iterable created using a function with a yield statement.
#The main feature of generator is evaluating the elements on demand. 
#In a function with a yield statement the state of the function is “saved” from the last call and can be picked up the next time you call a generator function.
def my_gen():
    for x in range(5):
        yield x
        
for x in my_gen():
    print(x)       
    
    
#GENERATOR EXPRESSIONS allow the creation of a generator on-the-fly without a yield keyword    
#The syntax and concept are similar to that of a list comprehension but GENERATOR_EXP uses () and list comprehension uses []
list_comp = [x ** 2 for x in range(10) if x % 2 == 0]
gen_exp = (x ** 2 for x in range(10) if x % 2 == 0)
print(list_comp)
#[0, 4, 16, 36, 64]
print(gen_exp)
#<generator object <genexpr> at 0x7f600131c410>  

#To get data from GENERATOR_EXP you need to loop into it
for x in gen_exp:
    print(x)  
    


#Convert list to comma seperated string
items=['foo','bar','ram']
print(','.join(items))

#convert number to comma seperated
items=[2,4,6,8,10]
print(','.join(map(str,items)))

from collections import Counter
a=[1,2,3,1,2,1,2,1,3,4,5,1,1,2]
cnt=Counter(a)
print(cnt.most_common(3))# # three most common elements


c = Counter('zabcdeabcdabcaba') 
sorted(c)  # list all unique elements in sorted fashion

c = Counter('abcdeabcdabcaba')
print(c)

sum(c.values()) # total of all counts

c['a'] # count of letter 'a'


for elem in 'shazamaaaaa':   # update counts from an iterable
    c[elem] += 1 

    
del c['b']  # remove all 'b'

c['a'] # count of letter 'a' is increased by shazamaaaa

d = Counter('simsalabim')       # make another counter
c.update(d)                     # add in the second counter
c['a']                          # now there are fourteen 'a'
c.clear()                       # empty the counter


a='abcdefghijklmnopqrstuvwyz'
print(a[::-1])   #reversing the string

for char in reversed(a):
    print(char)
    
num=123456789
print(int(str(num)[::-1]))   #reversing a number


#FOR ELSE
a=[1,2,3,4,5]
for el in a:
    if el ==5:
        break
else:
    print("Not executing break")    
    

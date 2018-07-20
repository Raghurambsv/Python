###################List comprehensions########################


#Find common numbers from two list using list comprehension
list_a = [1, 2, 3, 4]
list_b = [2, 3, 4, 5]

common=[a for a in list_a for b in list_b if a == b] # Give a or b before for (both gives same result)
print(common)

#Return numbers from the list which are not equal as tuple:
list_a = [1, 2, 3]
list_b = [2, 7]

different_num = [(a, b) for a in list_a for b in list_b if a != b]
print(different_num) # Output: [(1, 2), (1, 7), (2, 7), (3, 2), (3, 7)]     



#List comprehensions for iterating strings
list_a = ["Hello", "World", "In", "Python"]

small_list_a = [str.lower() for str in list_a]

print(small_list_a) # Output: ['hello', 'world', 'in', 'python']

#list comprehensions can be used to produce list of list
list_a = [1, 2, 3]

square_cube_list = [ [a**2, a**3] for a in list_a]

print(square_cube_list) # Output: [[1, 1], [4, 8], [9, 27]]


#Concatenating lists
list_a = [1, 2, 3, 4]
list_b = [5, 6, 7, 8]

list_c=list_a + list_b
print(list_c)
#or
list_a= list_a + list_b
print(list_a)
#or
list_a = [1, 2, 3, 4]
list_b = [5, 6, 7, 8]
list_c=list_a.extend(list_b)
print(list_c) #gives none object coz list_a is extended and so list_c just returns a NONE object
print(list_a) #will give the full concatenation of both the list_a & list_b


###################DECORATORS#######################

##python syntax for decorators
#@decorator_func
#def say_hello():
#    print ('Hello')
#
##The above statements is equivalent to    
#def say_hello():
#    print ('Hello')
#say_hello = deocrator_func(say_hello)    


#################Decorators Example
def decorator_func(arg): #Here arg is not a variable as usual ...but you are sending a Another_function as a argument
  # define another wrapper function which modifies some_func
  def wrapper_func():
    print("Wrapper function started")
    
    arg()
    
    print("Wrapper function ended")
    
  return wrapper_func # Wrapper function add something to the passed function and decorator returns the wrapper function
    
def say_hello():
  print ("Hello")
  
var1 = decorator_func(say_hello) #sending a function "say_hello" as a argument
var1()


################Using Python shortform for decorators "@" (Above example can be written as below also)
def decorator_func(arg):
    def wrapper_func():
        print("Wrapper function START")
        arg()
        print("Wrapper function END")
    
    return wrapper_func

@decorator_func
def say_hello():
    print("Hello")

say_hello()    
    

###################GLOBAL keyword#######################

#Normal func 
def add(value1, value2):
    return value1 + value2

result = add(3, 5) #You need to declare a variable explicitly to collect the RETURN VALUE from func
print(result)



def add(value1,value2):
    global result       #result is a global variable ...whose scope is outside the func too
    result = value1 + value2

add(3,5)
print(result)


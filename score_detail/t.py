import json

dict1 = {'a': '666'}
dict2 = {
    'b': '8756', 
    'c': {
        'cd': '9878', 
        'ce': 'ujde'
    }
}

dict1.update(dict2)
print(dict1)
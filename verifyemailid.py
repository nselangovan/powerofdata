import re
import csv
import pandas as pd

regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
valid_email_id_flag = '0'
my_str = []

def check(email):
    # pass the regular expression
    # and the string into the fullmatch() method
    if (re.fullmatch(regex, email)):
        print("Valid Email")
        valid_email_id_flag = '1'
        return valid_email_id_flag
    # print(valid_email_id_flag)
    else:
        print("Invalid Email")
        valid_email_id_flag = '2'
        return valid_email_id_flag


user_name = input('Enter the valid Email id : ')
valid_email_id_flag = check(user_name)

print('my val ', valid_email_id_flag)
if (valid_email_id_flag == '1'):
    first_name = input('Enter the first name : ')
    last_name = input('Enter the last name : ')
    password = input('Enter the password : ')
    email_id = user_name
# my_str = [[user_name],[first_name],[first_name],[password],[email_id]]
my_str = user_name # + "','" + first_name +"','"+first_name+"','"+password+"','"+email_id"'"
print(my_str)


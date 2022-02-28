import re
import csv
import sys
import pandas as pd

regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
valid_email_id_flag = '0'


def validate_password(password):
    l, u, p, d = 0, 0, 0, 0
    if (len(password) >= 8):
        for i in password:

            # counting lowercase alphabets
            if (i.islower()):
                l += 1

            # counting uppercase alphabets
            if (i.isupper()):
                u += 1

            # counting digits
            if (i.isdigit()):
                d += 1

            # counting the mentioned special characters
            if (i == '@' or i == '$' or i == '_'):
                p += 1
    if (l >= 1 and u >= 1 and p >= 1 and d >= 1 and l + p + u + d == len(password)):
        print("Valid Password")
    else:
        print("Invalid Password")
        sys.exit()

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
user_name =user_name.upper()
print('user name ' , user_name)
if (valid_email_id_flag == '1'):
    df = pd.read_csv('employee_file2.csv')
    lv_validate = list(df["username"])
    #print(lv_validate , type(lv_validate))
    if user_name in lv_validate:
        print('i am there already')
        #print('my val ', valid_email_id_flag)
    else:
        if (valid_email_id_flag == '1'):
            first_name = input('Enter the first name : ')
            last_name = input('Enter the last name : ')
            password = input('Enter the password : ')
            validate_password(password)
            email_id = user_name
            with open('employee_file2.csv', mode='a') as csv_file:
                fieldnames = ['username', 'firstname', 'lastname', 'email','password']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                #writer.writeheader()
                writer.writerow({'username': user_name, 'firstname': first_name, 'lastname': last_name , 'email': user_name , 'password':password})
else:
    print('Invalid Email id.')
#print(df["username"])

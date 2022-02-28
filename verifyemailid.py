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
            email_id = user_name
            with open('employee_file2.csv', mode='a') as csv_file:
                fieldnames = ['username', 'firstname', 'lastname', 'email','password']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                #writer.writeheader()
                writer.writerow({'username': user_name, 'firstname': first_name, 'lastname': last_name , 'email': user_name , 'password':password})
else:
    print('Invalid Email id.')
#print(df["username"])

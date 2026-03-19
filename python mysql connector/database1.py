import mysql.connector

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Chinnu@118'
)

if conn.is_connected():
    print('connection established')

    # Step 1: Create cursor
    mycursor = conn.cursor()

    # Step 2: Execute query
   
mycursor.execute("DROP DATABASE pythondb")
mycursor.execute("CREATE DATABASE pythondb")
print(mycursor)
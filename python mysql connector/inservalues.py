import mysql.connector

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Chinnu@118',
    database = 'pythondb'
)
mycursor = conn.cursor()

sql = 'insert into student(name, branch, id) values (%s, %s, %s)'

# if user want to create multiple values then we can create list

val =[('john', 'cse',' 56'), ('mike', 'IT', '28'), ('tyson', 'me', '80')] 

mycursor.executemany(sql,val)
conn.commit()
print(mycursor.rowcount, 'record inserted')


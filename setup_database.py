import sqlite3

def init_db():
    conn = sqlite3.connect('college_data.db')
    c = conn.cursor()

    # 1. Create Tables (Added 'dob' column)
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            usn TEXT PRIMARY KEY,
            name TEXT,
            dob TEXT,  -- Acts as the password
            sem INTEGER,
            internal1 REAL,
            internal2 REAL,
            absences INTEGER,
            failures INTEGER
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS proctorial (
            usn TEXT PRIMARY KEY,
            study_time INTEGER,
            health INTEGER,
            famrel INTEGER,
            goout INTEGER,
            freetime INTEGER,
            FOREIGN KEY(usn) REFERENCES students(usn)
        )
    ''')

    # 2. Insert Dummy Data with DOBs (Format: YYYY-MM-DD)
    # Student 1: Rahul (The High Performer)
    c.execute("INSERT OR REPLACE INTO students VALUES ('1RV23MCA001', 'Rahul Sharma', '2001-05-15', 4, 18, 19, 2, 0)")
    c.execute("INSERT OR REPLACE INTO proctorial VALUES ('1RV23MCA001', 4, 5, 5, 2, 3)")

    # Student 2: Anjali (The At-Risk Student)
    c.execute("INSERT OR REPLACE INTO students VALUES ('1RV23MCA002', 'Anjali Gupta', '2002-08-20', 4, 15, 14, 12, 0)")
    c.execute("INSERT OR REPLACE INTO proctorial VALUES ('1RV23MCA002', 2, 2, 3, 5, 4)")

    # Student 3: Karthik (The Critical Case)
    c.execute("INSERT OR REPLACE INTO students VALUES ('1RV23MCA003', 'Karthik R', '2001-12-10', 4, 8, 7, 25, 2)")
    c.execute("INSERT OR REPLACE INTO proctorial VALUES ('1RV23MCA003', 1, 3, 2, 5, 5)")

    conn.commit()
    conn.close()
    print("Database updated with Security Layer (DOB)!")

if __name__ == "__main__":
    init_db()
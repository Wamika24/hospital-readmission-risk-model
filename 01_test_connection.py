import psycopg2

print("=" * 50)
print("READMISSION AI PROJECT - DATABASE TEST")
print("=" * 50)

try:
    conn = psycopg2.connect(
        host="localhost",
        port="5433",
        database="readmission_db",
        user="postgres",
        password="1234"
    )

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM diabetes_clean;")
    count = cursor.fetchone()[0]

    print("Connected successfully!")
    print(f"Found {count} rows in diabetes_clean")

    cursor.close()
    conn.close()

except Exception as e:
    print("Connection failed:")
    print(e)

print("=" * 50)
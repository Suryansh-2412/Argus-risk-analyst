import sqlite3

conn = sqlite3.connect("data/finai.db")
cursor = conn.cursor()

print("=== TABLES ===")
tables = cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table';"
).fetchall()
print(tables)

print()

for table in ["articles", "nlp_scores", "daily_risk_scores"]:
    try:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
        print(f"{table}: {count} rows")
    except Exception as e:
        print(f"{table}: ERROR -> {e}")

conn.close()
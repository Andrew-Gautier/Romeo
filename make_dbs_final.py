from cwe_db import CVE_DB

devign_path = "C:\\Users\\Andrew\\Downloads\\function.json"
julietc_path = "C:\\Users\\Andrew\\OneDrive\\Documents\\Juliet C_C++ 1.3\\testcases"
julietjava_path = "C:\\Users\\Andrew\\OneDrive\\Documents\\Juliet Java 1.3\\src\\testcases"
julietcsharp_path = "C:\\Users\\Andrew\\OneDrive\\Documents\\Juliet C# 1.3\\src\\testcases"
bugs_path = "C:\\Users\\Andrew\\Desktop\\BugsInPy\\projects"


db = CVE_DB("devign.db")

db.cur.execute("PRAGMA optimize")
db.cur.execute("PRAGMA journal_mode = WAL")

print ("=== Creating Devign DB ===")

db.devign(devign_path).commit()
db.close()

db2 = CVE_DB("bugsinpy.db")

db2.cur.execute("PRAGMA optimize")
db2.cur.execute("PRAGMA journal_mode = WAL")

print ("=== Creating BugsInPy DB ===")

db2.bugsinpy(bugs_path).commit()

db2.close()
db3 = CVE_DB("juliet_c.db")

db3.cur.execute("PRAGMA optimize")
db3.cur.execute("PRAGMA journal_mode = WAL")

print ("=== Creating Juliet C DB ===")

db3.juliet(julietc_path, min_lines = 10).commit()
db3.close()

db4 = CVE_DB("juliet_java.db")

db4.cur.execute("PRAGMA optimize")
db4.cur.execute("PRAGMA journal_mode = WAL")

print ("=== Creating Juliet Java DB ===")

db4.juliet(julietjava_path, min_lines = 10).commit()
db4.close()

db5 = CVE_DB("juliet_csharp.db")

db5.cur.execute("PRAGMA optimize")
db5.cur.execute("PRAGMA journal_mode = WAL")

print ("=== Creating Juliet C# DB ===")

db5.juliet(julietcsharp_path, min_lines = 10).commit()
db5.close()

print("Finished successfully!!")

import cwe_db

java_source = "C:\\Users\\Andrew\\OneDrive\\Documents\\Juliet Java 1.3\\src\\testcases"
c_source = "C:\\Users\\Andrew\\OneDrive\\Documents\\Juliet C_C++ 1.3\\testcases"

java_manifest = "manifests\\java_manifest.xml"
c_manifest = "manifests\\cmanifest.xml"

cwe_db.record("java_10+.db", java_manifest, java_source, min_lines=10)
print("Java functions recorded successfully!")
cwe_db.record("c_10+.db", c_manifest, c_source, min_lines=10)
print("C functions recorded successfully!")
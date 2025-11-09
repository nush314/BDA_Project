from pyspark.sql import SparkSession
import pyspark

print("="*70)
print("TESTING PYSPARK INSTALLATION")
print("="*70)

print(f"\nPySpark version: {pyspark.__version__}")

# Create Spark session
print("\nCreating Spark session...")
spark = SparkSession.builder \
    .appName("TestPySpark") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("✓ Spark session created successfully!")
print(f"  Spark version: {spark.version}")
print(f"  Master: {spark.sparkContext.master}")

# Test basic operation
print("\nTesting basic Spark operation...")
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
df = spark.createDataFrame(data, ["name", "age"])

print("✓ DataFrame created:")
df.show()

print("\n✓ PySpark is working correctly!")
print("="*70)

spark.stop()
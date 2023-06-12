from pyspark.sql import SparkSession
from pyspark.sql.functions import col, stddev, mean, udf
from pyspark.sql.types import ArrayType, FloatType, IntegerType

if __name__ == '__main__':
    spark = SparkSession.builder.master("local[1]") \
        .appName('TestTask') \
        .getOrCreate()

    train_data = spark.read.csv('train.csv',header=True, inferSchema=True)

    test_data = spark.read.csv('test.csv',header=True, inferSchema=True)

    range_num = train_data.columns.__len__() - 2

    means = train_data.select([mean(col(f'feature_type_1_{i}')).alias(f'mean_{i}') for i in range(range_num)])

    stds = train_data.select([stddev(col(f'feature_type_1_{i}')).alias(f'std_{i}') for i in range(range_num)])

    test_with_stats = test_data.join(means).join(stds)

    standardize_udf = udf(lambda value, mean, std: float(value - mean) / std, FloatType())

    for i in range(range_num):
        feature_name = f'feature_type_1_{i}'

        stand_feature_name = f'features_type_1_stand_{i}'

        test_with_stats = test_with_stats.withColumn(stand_feature_name,
                                                     standardize_udf(col(feature_name), col(f'mean_{i}'),
                                                                     col(f'std_{i}')))


    max_index_udf = udf(lambda *values: values.index(max(values)), IntegerType())

    test_data = test_with_stats.withColumn('max_feature_type_1_index',
                                           max_index_udf(*[col(f'feature_type_1_{i}') for i in range(range_num)]))

    df = test_data.select('id_job', *[f'features_type_1_stand_{i}' for i in range(range_num)],
                          'max_feature_type_1_index')

    df.write.option("header", True).csv('test_transformed.csv')
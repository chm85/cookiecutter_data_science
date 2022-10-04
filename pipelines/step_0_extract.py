# Databricks notebook source
import logging
import numpy as np
import mlflow
from pyspark.shell import spark
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# Source Connections
JDBC_CONNECTION = ''
DB_CONN = conn_str()
exec(DB_CONN)

# Run Name
SOURCE_SCRIPT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[-1]

# Data bricks table names
DB_CMD = constants[SOURCE_SCRIPT]['Database']
spark.sql(DB_CMD)
TBL_NAME = 'step_0_dataset'

if __name__ == "__main__":

    mlflow.set_experiment("/Users/cmooney@carhartt.com/D2C_Store_Classifier")
    with mlflow.start_run(run_name=SOURCE_SCRIPT):

        # Connect to Datastorage
        with open("../sql/consumer_sales.sql", "r", encoding='utf-8') as file_handle:
            CONSUMER_SQL = file_handle.read()

        # Read Data
        dataset = (
            spark.read
            .format("jdbc")
            .option("url", JDBC_CONNECTION)
            .option("query", CONSUMER_SQL)
            .load()
        ).toPandas().fillna(0)

        # Normalize columns
        dataset = CustomerData.clean_columns(dataset=dataset)
        dataset.columns = [x.lower() for x in dataset.columns]

        # fill na to avoid pyspark error
        dataset = dataset.fillna(0)

        # Fixing issues where pyspark gets confused with longtypes in string cols
        dataset['gender'] = np.where(dataset['gender'].isin(['M', 'F']), dataset['gender'], 'UNK')
        dataset['color'] = np.where(dataset['color'] == 0, 'UNK', dataset['color'])
        dataset['category'] = np.where(dataset['category'] == 0, 'UNK', dataset['category'])

        # Write to databricks
        dataset = spark.createDataFrame(dataset)
        dataset.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TBL_NAME)

    mlflow.log_param("dataset count", dataset.count())
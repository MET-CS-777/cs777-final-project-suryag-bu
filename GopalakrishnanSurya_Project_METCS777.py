from __future__ import print_function

import sys
import time
from pyspark.sql import SparkSession

import pandas as pd
from pyspark.sql import functions as F 
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import when, col
from pyspark.sql import functions as F, Window


'''
This script performs Lasso Regression using PySpark on a healthcare dataset.
It builds a full end-to-end ML pipeline:
 - Data loading, cleaning, and transformation
 - Rare category collapsing
 - Feature encoding (StringIndexer + OneHotEncoder)
 - Vector assembling
 - Lasso regression model training
 - Automatic tuning of regularization parameter (regParam)
 - Model evaluation with R² and RMSE

It is designed to improve interpretability and avoid overfitting
by shrinking less-important coefficients to zero (L1 regularization).
'''


def collapse_rare_categories(df, colname, threshold=25):
    """
    Cluster-safe version: Collapse rare categories (< threshold) into 'Other'
    without using join or groupBy (avoids shuffles).
    Example:
    If 'State' has 50 categories but only 2 appear <15 times,
    those 2 will be replaced with 'Other'.
    """

    # Compute category frequency using window over the same partition
    freq_col = f"{colname}_freq"
    w = Window.partitionBy(colname)

    # Add frequency count column to the dataframe
    df = df.withColumn(freq_col, F.count(colname).over(w)) #Calculates the count of how many rows are in that partition

    # Replace rare categories. Keep original value "TEXAS" if count>=25
    df = df.withColumn(
        colname,
        F.when(F.col(freq_col) >= threshold, F.col(colname)).otherwise(F.lit("Other"))
    ).drop(freq_col)

    return df



if __name__ == "__main__":
    # ---------- Spark Session ----------
    # Initialize Spark environment
    spark = SparkSession.builder.appName("TermProject").getOrCreate()
    sc = spark.sparkContext
    spark.sparkContext.setLogLevel("ERROR")  # Reduce console verbosity

    # ---------- Load CSV from command-line argument ----------
    # sys.argv[1] = path to the CSV file passed when running the script
    dataset = spark.read.csv(sys.argv[1], header=True, inferSchema=True)
    
    # ---------- Select relevant columns ----------
    # We’ll only use the following columns for modeling
    selected_cols = [
        "State",
        "Domain",
        "Reporting Program",
        "Measure Abbreviation",
        "Measure Type",
        "Population",
        "Methodology",
        "Core Set Year",
        "State Rate"  # This is the target variable (Y)
    ]
    dataset = dataset.select(selected_cols)

    # ---------- Data Cleaning ----------
    # Convert Core Set Year and State Rate to numeric (double)
    # Non-numeric values will be replaced with None (null)
    dataset = dataset.withColumn(
        "Core Set Year",
        when(col("Core Set Year").rlike("^[0-9.]+$"), col("Core Set Year").cast("double")).otherwise(None)
    )

    dataset = dataset.withColumn(
        "State Rate",
        when(col("State Rate").rlike("^[0-9.]+$"), col("State Rate").cast("double")).otherwise(None)
    )

    # ---------- Sanity Check ----------
    dataset.printSchema()
    dataset.show(2, truncate=False)

    # ---------- Split into Train/Test ----------
    # Randomly split 80% for training, 20% for testing
    train_df, test_df = dataset.randomSplit([0.8, 0.2], seed=42)

    print("Train count:", train_df.count())
    print("Test count:", test_df.count())

    # ---------- Drop Rows with Missing Target/Year ----------
    # Remove rows where either target (State Rate) or Core Set Year is missing
    train_df = train_df.na.drop(subset=["State Rate", "Core Set Year"])
    test_df = test_df.na.drop(subset=["State Rate", "Core Set Year"])

    print("Train count after dropping NULLs:", train_df.count())
    print("Test count after dropping NULLs:", test_df.count())

    # ---------- Define Features ----------
    # Categorical features to encode
    categorical_cols = [
        "State", "Domain", "Reporting Program",
        "Measure Abbreviation", "Measure Type",
        "Population", "Methodology"
    ]
    # Collapse rare categories to "Other"
    for c in categorical_cols:
        dataset = collapse_rare_categories(dataset, c, threshold=25)

    # Numeric features (directly used)
    numeric_cols = ["Core Set Year"]

    # ---------- Feature Encoding ----------
    # StringIndexer converts categorical text columns to numeric indices
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") 
                for c in categorical_cols]

    # OneHotEncoder converts indices to binary vectors (one-hot encoding)
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec") 
                for c in categorical_cols]
    
    # Combine all feature vectors (encoded categorical + numeric)
    assembler_inputs = [f"{c}_vec" for c in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    
    # ---------- Lasso Regression Training ----------
    # We will loop through different regularization strengths (regParam)
    # regParam controls penalty strength (higher = more shrinkage)
    # elasticNetParam=1.0 means pure L1 regularization (Lasso)
    reg_params = [0.01, 0.1, 0.2, 0.5]
  

    for rp in reg_params:
        # Define model
        lasso = LinearRegression(
            featuresCol="features",
            labelCol="State Rate",
            regParam=rp,          # regularization strength (λ)
            elasticNetParam=1.0   # 1.0 = Lasso, 0.0 = Ridge
        )

        # Combine preprocessing and model training in one pipeline
        stages = indexers + encoders + [assembler, lasso]
        pipeline = Pipeline(stages=stages)

        # Fit pipeline on training data
        model = pipeline.fit(train_df)

        # Predict on test data
        predictions = model.transform(test_df)

        # ---------- Evaluation ----------
        # Compute R² and RMSE to measure model performance
        # State rate is the target col, prediction has model predicted values
        evaluator_r2 = RegressionEvaluator(
            labelCol="State Rate", predictionCol="prediction", metricName="r2"
        )
        evaluator_rmse = RegressionEvaluator(
            labelCol="State Rate", predictionCol="prediction", metricName="rmse"
        )

        r2 = evaluator_r2.evaluate(predictions)
        rmse = evaluator_rmse.evaluate(predictions)

        # ---------- Display Results ----------
        print("\n================")
        print(" Model Performance ")
        print("==================")
        print("Regularization Parameter ", rp)
        print(f"R² Score : {r2:.4f}")  # R² closer to 1.0 means better fit
        print(f"RMSE     : {rmse:.4f}")  # RMSE lower = better predictive accuracy

    # ---------- Inspect Lasso Coefficients ----------
        # Extract the trained LinearRegressionModel 
        lasso_model = model.stages[-1]

        # Get numeric coefficients
        coefficients = lasso_model.coefficients.toArray()  # numpy array

        # Feature names are the same as assembler_inputs
        feature_names = assembler_inputs

        # Combine feature names with coefficients
        feature_importance = list(zip(feature_names, coefficients))

        # Keep only non-zero coefficients (features selected by Lasso)
        important_features = [(name, coef) for name, coef in feature_importance if coef != 0]

        # Sort by absolute coefficient value descending
        important_features = sorted(important_features, key=lambda x: abs(x[1]), reverse=True)

        # Display important features
        print("\n================ Important Features (Non-zero Lasso Coefficients) ================")
        for name, coef in important_features:
            print(f"{name}: {coef:.4f}")



    spark.stop()

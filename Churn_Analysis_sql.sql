-- ============================================================
-- Customer Churn Analysis — SQL Business Queries
-- Tool: MySQL Workbench
-- ============================================================

-- Create Database
CREATE DATABASE churn_analysis;
USE churn_analysis;

-- Create Table
CREATE TABLE churn_data (
    CustomerID varchar(50),
    Gender varchar(20),
    SeniorCitizen int,
    Partner varchar(10),
    Dependents varchar(10),
    Tenure int,
    PhoneService varchar(10),
    MultipleLines varchar(50),
    InternetService varchar(50),
    OnlineSecurity varchar(50),
    OnlineBackup varchar(50),
    DeviceProtection varchar(50),
    TechSupport varchar(50),
    StreamingTV varchar(50),
    StreamingMovies varchar(50),
    Contract varchar(50),
    PaperlessBilling varchar(10),
    PaymentMethod varchar(100),
    MonthlyCharges float,
    TotalCharges float,
    Churn varchar(10)
);

-- ------------------------------------------------------------
-- Data was loaded using MySQL Workbench's Table Data Import
-- Wizard (Schema -> churn_data -> right click -> Table Data
-- Import Wizard) pointing at cleaned_churn_data.csv.
-- If you'd rather load it from the CLI instead, run:
--   LOAD DATA LOCAL INFILE 'data/cleaned_churn_data.csv'
--   INTO TABLE churn_data
--   FIELDS TERMINATED BY ','
--   ENCLOSED BY '"'
--   LINES TERMINATED BY '\n'
--   IGNORE 1 ROWS;
-- ------------------------------------------------------------

-- Sanity check
SELECT * FROM churn_data LIMIT 100;
SELECT COUNT(*) FROM churn_data;

-- ============================================================
-- Business Queries
-- ============================================================

-- 1. Total Customers
SELECT COUNT(*) AS TotalCustomers
FROM churn_data;

-- 2. Total Churned Customers
SELECT COUNT(*) AS ChurnedCustomers
FROM churn_data
WHERE Churn = 'Yes';

-- 3. Churn By Contract Type
SELECT Contract, COUNT(*) AS ChurnCount
FROM churn_data
WHERE Churn = 'Yes'
GROUP BY Contract
ORDER BY ChurnCount DESC;

-- 4. Payment Method Churn
SELECT PaymentMethod, COUNT(*) AS ChurnCustomers
FROM churn_data
WHERE Churn = 'Yes'
GROUP BY PaymentMethod
ORDER BY ChurnCustomers DESC;

-- 5. Average Monthly Charges by Churn Status
SELECT Churn, AVG(MonthlyCharges) AS AvgMonthlyCharges
FROM churn_data
GROUP BY Churn;

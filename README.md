# Project Architecture

## Overview

This project follows a Lakehouse Architecture to analyze FIFA World Cup 2026 data.

The project was developed to demonstrate Data Analytics, Data Science and Data Engineering skills using Python, SQL, Databricks and Power BI.

## Business Goal

Estimate which national teams have the highest probability of advancing through the FIFA World Cup 2026 stages.

## Data Sources

* OpenFootball
* Football-Data
* Kaggle World Cup Datasets

## Architecture Layers

### Raw Layer

Stores original data exactly as received from external sources.

### Bronze Layer

Stores structured raw data without business transformations.

### Silver Layer

Applies cleaning, validation, standardization and business rules.

### Gold Layer

Contains analytical tables ready for reporting, dashboards and machine learning.

## Data Flow

Data Sources
→ Python Ingestion
→ Raw Layer
→ Bronze Layer
→ Silver Layer
→ Gold Layer
→ SQL Analytics
→ Machine Learning
→ Power BI Dashboard
→ Business Insights

## Technologies

* Python
* Pandas
* SQL
* Databricks
* Delta Lake
* Git
* GitHub
* Power BI

## Expected Deliverables

* Data Pipeline
* Data Quality Checks
* Analytical Tables
* SQL Queries
* Machine Learning Model
* Power BI Dashboard
* Technical Documentation

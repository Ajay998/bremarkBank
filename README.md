# BreMark Bank Credit Card Launch: Phase 1 Analysis

## Project Overview

This project analyzes customer transactions and credit profiles to identify the optimal target group for BreMark Bank's new credit card launch. The analysis uses data science techniques to clean, process, and analyze customer data to provide strategic insights for the credit card marketing campaign.

## Business Objective

**Primary Goal:** Identify and analyze customer segments to determine the best target group for BreMark Bank's credit card launch, focusing on untapped markets with growth potential.

**Key Questions Answered:**
- Which customer age group has the highest potential for credit card adoption?
- What are the spending patterns across different demographics?
- Which customer segments are underutilizing credit card services?
- What product categories and platforms do target customers prefer?

## Dataset Description

The analysis uses three main datasets:

### 1. Customers Dataset (`customers.csv`)
- **cust_id**: Unique customer identifier
- **age**: Customer age
- **gender**: Customer gender
- **location**: Customer location
- **occupation**: Customer occupation
- **marital_status**: Marital status
- **annual_income**: Annual income in dollars

### 2. Credit Profiles Dataset (`credit_profiles.csv`)
- **cust_id**: Unique customer identifier
- **credit_score**: Customer credit score (300-800)
- **credit_limit**: Maximum credit limit available
- **credit_utilisation**: Current credit utilization percentage
- **outstanding_debt**: Current outstanding debt amount

### 3. Transactions Dataset (`transactions.csv`)
- **cust_id**: Unique customer identifier
- **tran_amount**: Transaction amount
- **product_category**: Category of purchased product
- **platform**: Shopping platform used
- **payment_type**: Payment method used

## Key Features

### Data Cleaning & Preprocessing
- **Null Value Handling**: Occupation-wise median imputation for missing income and age data
- **Outlier Detection & Treatment**: Statistical and business rule-based outlier identification
- **Duplicate Removal**: Systematic duplicate record cleaning
- **Data Validation**: Business logic validation for credit limits and transaction amounts

### Advanced Analytics
- **Age Group Segmentation**: Customer classification into strategic age brackets
- **Income Analysis**: Occupation and demographic-based income analysis
- **Credit Profile Analysis**: Credit score, limit, and utilization assessment
- **Transaction Pattern Analysis**: Spending behavior and payment preference analysis
- **Correlation Analysis**: Multi-variable relationship identification

### Visualization & Insights
- **Distribution Analysis**: Age, income, and credit score distributions
- **Comparative Analysis**: Cross-demographic comparisons
- **Payment Pattern Visualization**: Credit card vs. other payment methods
- **Platform & Category Analysis**: Shopping behavior insights

## Installation & Setup

### Prerequisites
```bash
# Python 3.7 or higher
python --version

# pip package manager
pip --version
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Data Setup
1. Create a `data` folder in the project root directory
2. Place the following CSV files in the `data` folder:
   - `customers.csv`
   - `credit_profiles.csv`
   - `transactions.csv`

### Project Structure
```
bremark-bank-analysis/
├── bremarchPhaseOne.py          # Main analysis script
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── data/                       # Data directory
    ├── customers.csv           # Customer demographics data
    ├── credit_profiles.csv     # Credit information data
    └── transactions.csv        # Transaction history data
```

## Usage

### Run Complete Analysis
```bash
python bremarchPhaseOne.py
```

### Run with MySQL (Optional)
If you have MySQL database setup:
```python
# Update MySQL credentials in load_data_from_mysql() function
conn = mysql.connector.connect(
    host='your_host',
    user='your_username',  
    passwd='your_password',
    database='your_database'
)
```

## Key Findings & Recommendations

### Target Market Identification: Age Group 18-25

**Market Characteristics:**
- Represents ~26% of total customer base
- Average annual income <$50K
- Limited credit history and lower credit scores
- Lower credit card adoption compared to other age groups
- High engagement in Electronics, Fashion & Apparel, and Beauty categories

**Strategic Recommendations:**
1. **Primary Target**: Focus credit card launch on 18-25 age group
2. **Product Categories**: Emphasize rewards for Electronics, Fashion, and Beauty purchases
3. **Platform Strategy**: Leverage Amazon, Flipkart, and Alibaba partnerships
4. **Credit Limit Strategy**: Offer appropriate limits based on income and credit history
5. **Marketing Approach**: Position as starter credit card for building credit history

### Business Impact
- **Market Opportunity**: Untapped segment with 26% market share
- **Growth Potential**: Young customers with increasing income trajectory
- **Long-term Value**: Early relationship building for lifetime customer value
- **Competitive Advantage**: First-mover advantage in underserved segment

## Technical Implementation

### Data Processing Pipeline
1. **Data Loading**: CSV/MySQL data ingestion
2. **Data Cleaning**: Null handling, outlier treatment, duplicate removal
3. **Feature Engineering**: Age grouping, income categorization
4. **Analysis**: Statistical analysis and correlation studies
5. **Visualization**: Comprehensive plotting and charting
6. **Insights Generation**: Business recommendations

### Key Algorithms Used
- **Outlier Detection**: 3-sigma rule and IQR method
- **Imputation**: Median-based null value replacement
- **Segmentation**: Statistical binning for age groups
- **Correlation Analysis**: Pearson correlation for numerical variables

## Performance Metrics

### Data Quality Metrics
- **Completeness**: 100% after null value treatment
- **Consistency**: Validated business rules applied
- **Accuracy**: Outlier treatment maintains data integrity

### Analysis Coverage
- **Customer Demographics**: 100% coverage across all segments
- **Credit Profiles**: Complete credit assessment
- **Transaction Analysis**: Full transaction pattern analysis




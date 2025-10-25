# BreMark Bank Credit Card Launch: Complete Analysis

## Project Overview

This comprehensive project analyzes customer transactions and credit profiles to identify the optimal target group for BreMark Bank's new credit card launch, followed by A/B testing to validate the effectiveness of the new credit card product. The analysis uses advanced data science techniques, statistical hypothesis testing, and A/B testing methodologies to provide strategic insights for the credit card marketing campaign.

## Business Objective

**Primary Goal:** Identify and analyze customer segments to determine the best target group for BreMark Bank's credit card launch, then validate the product's effectiveness through controlled A/B testing.

**Key Questions Answered:**
- Which customer age group has the highest potential for credit card adoption?
- What are the spending patterns across different demographics?
- Which customer segments are underutilizing credit card services?
- What product categories and platforms do target customers prefer?
- Does the new credit card increase average transaction amounts significantly?
- Is the new credit card product launch justified by statistical evidence?

## Project Phases

### Phase 1: Market Segmentation & Target Identification
Comprehensive customer data analysis to identify the optimal target market segment.

### Phase 2: A/B Testing & Validation
Statistical hypothesis testing to validate the effectiveness of the new credit card in increasing transaction amounts.

## Dataset Description

### Phase 1 Datasets

#### 1. Customers Dataset (`customers.csv`)
- **cust_id**: Unique customer identifier
- **age**: Customer age
- **gender**: Customer gender
- **location**: Customer location
- **occupation**: Customer occupation
- **marital_status**: Marital status
- **annual_income**: Annual income in dollars

#### 2. Credit Profiles Dataset (`credit_profiles.csv`)
- **cust_id**: Unique customer identifier
- **credit_score**: Customer credit score (300-800)
- **credit_limit**: Maximum credit limit available
- **credit_utilisation**: Current credit utilization percentage
- **outstanding_debt**: Current outstanding debt amount

#### 3. Transactions Dataset (`transactions.csv`)
- **cust_id**: Unique customer identifier
- **tran_amount**: Transaction amount
- **product_category**: Category of purchased product
- **platform**: Shopping platform used
- **payment_type**: Payment method used

### Phase 2 Datasets

#### 4. A/B Test Results (`avg_transactions_after_campaign.csv`)
- **control_group_avg_tran**: Daily average transaction amounts for control group (existing credit card)
- **test_group_avg_tran**: Daily average transaction amounts for test group (new credit card)
- **62 days** of post-campaign transaction data

## Key Features

### Phase 1: Data Analysis & Segmentation

#### Data Cleaning & Preprocessing
- **Null Value Handling**: Occupation-wise median imputation for missing income and age data
- **Outlier Detection & Treatment**: Statistical and business rule-based outlier identification
- **Duplicate Removal**: Systematic duplicate record cleaning
- **Data Validation**: Business logic validation for credit limits and transaction amounts

#### Advanced Analytics
- **Age Group Segmentation**: Customer classification into strategic age brackets (18-25, 26-48, 49-65)
- **Income Analysis**: Occupation and demographic-based income analysis
- **Credit Profile Analysis**: Credit score, limit, and utilization assessment
- **Transaction Pattern Analysis**: Spending behavior and payment preference analysis
- **Correlation Analysis**: Multi-variable relationship identification

#### Visualization & Insights
- **Distribution Analysis**: Age, income, and credit score distributions
- **Comparative Analysis**: Cross-demographic comparisons
- **Payment Pattern Visualization**: Credit card vs. other payment methods
- **Platform & Category Analysis**: Shopping behavior insights

### Phase 2: A/B Testing & Statistical Validation

#### Pre-Campaign Planning
- **Sample Size Calculation**: Statistical power analysis (α=0.05, power=0.8)
- **Effect Size Determination**: Analysis of required effect sizes (0.1 to 1.0)
- **Group Formation**: Control and test group selection (100 customers each)
- **Campaign Design**: 2-month trial period (09-10-23 to 11-10-23)

#### Hypothesis Testing
- **Null Hypothesis (H0)**: New credit card does NOT increase avg transaction amounts (μ_test ≤ μ_control)
- **Alternative Hypothesis (H1)**: New credit card DOES increase avg transaction amounts (μ_test > μ_control)
- **Test Type**: Right-tailed (one-sided) two-sample Z-test
- **Significance Level**: α = 0.05 (95% confidence)

#### Statistical Methods
- **Method 1**: Rejection Region Approach (Critical Z-value comparison)
- **Method 2**: P-value Approach (Manual calculation)
- **Method 3**: Statsmodels API (Validation using `sm.stats.ztest()`)

#### Validation Metrics
- **Descriptive Statistics**: Mean, standard deviation for both groups
- **Z-statistic**: Calculated test statistic
- **P-value**: Statistical significance measurement
- **Effect Size**: Practical significance assessment

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

#### Option 1: CSV Files
1. Create a `data` folder in the project root directory
2. Place the following CSV files in the `data` folder:
   - `customers.csv`
   - `credit_profiles.csv`
   - `transactions.csv`
   - `avg_transactions_after_campaign.csv` (for Phase 2)

#### Option 2: MySQL Database
1. Install MySQL on your localhost
2. Import the database using the SQL dump file:
```bash
mysql -u root -p < data/E_MasterCardDump.sql
```
3. Verify the import:
```bash
mysql -u root -p
USE e_master_card;
SHOW TABLES;
```

### Project Structure
```
bremark-bank-analysis/
├── bremarchPhaseOne.py          # Phase 1: Market segmentation analysis
├── bremarchPhaseTwo.py          # Phase 2: A/B testing analysis
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── data/                        # Data directory
    ├── customers.csv            # Customer demographics data
    ├── credit_profiles.csv      # Credit information data
    ├── transactions.csv         # Transaction history data
    ├── avg_transactions_after_campaign.csv  # A/B test results
    └── E_MasterCardDump.sql     # MySQL database dump (optional)
```

## Usage

### Phase 1: Run Market Segmentation Analysis
```bash
python bremarchPhaseOne.py
```

This will:
- Load and clean customer, credit profile, and transaction data
- Perform outlier detection and treatment
- Generate age group segmentation
- Create comprehensive visualizations
- Identify the 18-25 age group as the target market

### Phase 2: Run A/B Testing Analysis
```bash
python bremarchPhaseTwo.py
```

This will:
- Calculate required sample size based on statistical power
- Load post-campaign A/B test results
- Visualize distribution comparisons
- Perform three methods of hypothesis testing
- Provide statistical validation and business recommendations

### Run with MySQL (Optional)
If you have MySQL database setup:
```python
# Update MySQL credentials in load_data_from_mysql() function
conn = mysql.connector.connect(
    host='localhost',
    user='root',  
    passwd='root',
    database='e_master_card'
)
```

## Key Findings & Recommendations

### Phase 1: Target Market Identification

#### Target Market: Age Group 18-25

**Market Characteristics:**
- Represents ~26% of total customer base (~246 customers)
- Average annual income <$50K
- Limited credit history and lower credit scores
- Lower credit card adoption compared to other age groups
- High engagement in Electronics, Fashion & Apparel, and Beauty categories
- Lower average transaction amounts with credit cards

**Strategic Recommendations:**
1. **Primary Target**: Focus credit card launch on 18-25 age group
2. **Product Categories**: Emphasize rewards for Electronics, Fashion, and Beauty purchases
3. **Platform Strategy**: Leverage Amazon, Flipkart, and Alibaba partnerships
4. **Credit Limit Strategy**: Offer appropriate limits based on income and credit history
5. **Marketing Approach**: Position as starter credit card for building credit history

### Phase 2: A/B Testing Results

#### Statistical Findings

**Test Configuration:**
- **Control Group**: 40 customers using existing credit cards
- **Test Group**: 40 customers using new credit cards
- **Test Duration**: 62 days (2 months)
- **Sample Selection**: From 100 customers approached, 40% conversion rate

**Statistical Results:**
- **Z-statistic**: Significantly greater than critical value
- **P-value**: < 0.05 (statistically significant)
- **Decision**: REJECT null hypothesis
- **Conclusion**: New credit card DOES significantly increase average transaction amounts

**Business Impact:**
- **Average Transaction Increase**: Statistically significant improvement
- **Confidence Level**: 95% (α = 0.05)
- **Practical Significance**: Effect size of 0.4 standard deviations
- **Revenue Impact**: Measurable increase in per-transaction revenue

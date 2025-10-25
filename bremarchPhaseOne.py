"""
BreMark Bank Credit Card Launch: Phase 1
Analyze customers' transactions and credit profiles to figure out a target group 
for the launch of BreMark bank credit card
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def load_data_from_csv():
    """Load data from CSV files"""
    print("BREMARK BANK CREDIT CARD LAUNCH: PHASE 1")
    print("="*60)
    print("Objective: Analyze customers' transactions and credit profiles")
    print("to figure out a target group for the launch of BreMark bank credit card")

    print("\n" + "="*50)
    print("DATA IMPORT FROM CSV")
    print("="*50)
    
    df_cust = pd.read_csv('data/customers.csv')
    df_cs = pd.read_csv('data/credit_profiles.csv')
    df_trans = pd.read_csv('data/transactions.csv')
    
    print(f"Customers data: {df_cust.shape}")
    print(f"Credit Score data: {df_cs.shape}")
    print(f"Transactions data: {df_trans.shape}")
    
    return df_cust, df_cs, df_trans

def load_data_from_mysql():
    """Alternative method to load data from MySQL database"""
    try:
        import mysql.connector
        
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            passwd='root',
            database='e_master_card'
        )
        
        df_cust = pd.read_sql("SELECT * FROM customers", conn)
        df_trans = pd.read_sql("SELECT * FROM transactions", conn)
        df_cs = pd.read_sql("SELECT * FROM credit_profiles", conn)
        
        conn.close()
        
        print("Data loaded from MySQL successfully")
        return df_cust, df_cs, df_trans
    
    except Exception as e:
        print(f"MySQL connection failed: {e}")
        print("Falling back to CSV loading...")
        return load_data_from_csv()

def explore_customers_data(df_cust):
    """Explore and clean customers data"""
    print("\n" + "="*50)
    print("EXPLORE CUSTOMERS TABLE")
    print("="*50)
    
    print("First 3 rows:")
    print(df_cust.head(3))
    print(f"\nDataset description:")
    print(df_cust.describe())
    
    return df_cust

def handle_income_null_values(df_cust):
    """Handle null values in annual income column"""
    print("\n" + "="*40)
    print("1. ANALYZE INCOME COLUMN")
    print("="*40)
    
    print("Checking for null values:")
    print(df_cust.isnull().sum())
    
    if df_cust.annual_income.isnull().sum() > 0:
        print(f"\n{df_cust.annual_income.isnull().sum()} null values found in annual_income")
        print("Sample null records:")
        print(df_cust[df_cust.annual_income.isna()].head(4))
        
        # Calculate occupation-wise median income
        occupation_wise_inc_median = df_cust.groupby("occupation")["annual_income"].median()
        print(f"\nOccupation-wise median income:")
        print(occupation_wise_inc_median)
        
        # Replace null values with occupation-wise median
        df_cust['annual_income'] = df_cust.apply(
            lambda row: occupation_wise_inc_median[row['occupation']] if pd.isnull(row['annual_income']) else row['annual_income'],
            axis=1
        )
        
        print(f"\nNull values after treatment:")
        print(df_cust.isnull().sum())
    
    return df_cust, occupation_wise_inc_median

def detect_and_treat_income_outliers(df_cust, occupation_wise_inc_median):
    """Detect and treat outliers in annual income"""
    print("\n" + "="*40)
    print("OUTLIER DETECTION: ANNUAL INCOME")
    print("="*40)
    
    # Income distribution visualization
    plt.figure(figsize=(5, 5))
    sns.histplot(df_cust['annual_income'], kde=True, color='green', label='Data')
    plt.title('Histogram of annual_income')
    plt.show()
    
    print("Income statistics:")
    print(df_cust.annual_income.describe())
    
    # Statistical outlier detection using 3 standard deviations
    mean_income = df_cust['annual_income'].mean()
    std_income = df_cust['annual_income'].std()
    lower = mean_income - 3 * std_income
    upper = mean_income + 3 * std_income
    
    print(f"\nStatistical outlier boundaries: {lower:.2f} to {upper:.2f}")
    
    # Upper outliers
    upper_outliers = df_cust[df_cust['annual_income'] > upper]
    print(f"Upper outliers (>{upper:.0f}):")
    print(upper_outliers)
    
    # Business rule: minimum income should be 100
    lower_outliers = df_cust[df_cust.annual_income < 100]
    print(f"\nLower outliers (<100):")
    print(lower_outliers)
    
    # Treat lower outliers by replacing with occupation-wise median
    for index, row in df_cust.iterrows():
        if row["annual_income"] < 100:
            occupation = df_cust.at[index, "occupation"]
            df_cust.at[index, "annual_income"] = occupation_wise_inc_median[occupation]
    
    print(f"\nOutliers after treatment:")
    print(df_cust[df_cust.annual_income < 100])
    
    return df_cust

def visualize_income_analysis(df_cust):
    """Visualize income analysis by different categories"""
    print("\n" + "="*40)
    print("DATA VISUALIZATION: ANNUAL INCOME")
    print("="*40)
    
    # Average income per occupation
    avg_income_per_occupation = df_cust.groupby("occupation")["annual_income"].mean()
    
    plt.figure(figsize=(8,4))
    sns.barplot(x=avg_income_per_occupation.index, y=avg_income_per_occupation.values, palette='tab10')
    plt.xticks(rotation=45)
    plt.title('Average Annual Income Per Occupation')
    plt.xlabel('Occupation')
    plt.ylabel('Average Annual Income ($)')
    plt.show()
    
    # Income analysis by multiple categories
    cat_cols = ['gender', 'location', 'occupation', 'marital_status']
    num_rows = 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))
    axes = axes.flatten()
    
    for i, cat_col in enumerate(cat_cols):
        avg_income_by_category = df_cust.groupby(cat_col)['annual_income'].mean().reset_index()
        sorted_data = avg_income_by_category.sort_values(by='annual_income', ascending=False)
        
        sns.barplot(x=cat_col, y='annual_income', data=sorted_data, ci=None, ax=axes[i], palette='tab10')
        axes[i].set_title(f'Average Annual Income by {cat_col}')
        axes[i].set_xlabel(cat_col)
        axes[i].set_ylabel('Average Annual Income')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

def handle_age_outliers(df_cust):
    """Handle outliers in age column"""
    print("\n" + "="*40)
    print("2. ANALYZE AGE COLUMN")
    print("="*40)
    
    print("Age null values:")
    print(df_cust.age.isnull().sum())
    
    print("\nAge statistics:")
    print(df_cust.describe())
    
    min_age = df_cust.age.min()
    max_age = df_cust.age.max()
    print(f"Age range: {min_age} to {max_age}")
    
    # Age distribution visualization
    plt.hist(df_cust.age, bins=20, edgecolor='black')
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Customer Age Distribution")
    plt.axvline(min_age, color="red", label=f"Min Age: {min_age}")
    plt.axvline(max_age, color="green", label=f"Max Age: {max_age}")
    plt.legend()
    plt.show()
    
    # Find outliers
    outliers = df_cust[(df_cust.age < 15) | (df_cust.age > 80)]
    print(f"\nAge outliers (<15 or >80):")
    print(outliers)
    print(f"Total outliers: {outliers.shape[0]}")
    
    # Calculate median age per occupation for replacement
    median_age_per_occupation = df_cust.groupby('occupation')['age'].median()
    print(f"\nMedian age per occupation:")
    print(median_age_per_occupation)
    
    # Replace outliers with occupation-wise median
    for index, row in outliers.iterrows():
        if pd.notnull(row['age']):
            occupation = df_cust.at[index, 'occupation']
            df_cust.at[index, 'age'] = median_age_per_occupation[occupation]
    
    print(f"\nOutliers after treatment:")
    print(df_cust[(df_cust.age < 15) | (df_cust.age > 80)])
    
    return df_cust

def create_age_groups(df_cust):
    """Create age groups and visualize distribution"""
    print("\n" + "="*40)
    print("DATA VISUALIZATION: AGE COLUMN")
    print("="*40)
    
    # Define age groups
    bin_edges = [17, 25, 48, 65]
    bin_labels = ['18-25', '26-48', '49-65']
    
    df_cust['age_group'] = pd.cut(df_cust['age'], bins=bin_edges, labels=bin_labels)
    
    print("Age group distribution:")
    age_distribution = df_cust['age_group'].value_counts(normalize=True) * 100
    print(age_distribution)
    
    # Pie chart visualization
    plt.figure(figsize=(4, 4))
    plt.pie(
        age_distribution, 
        labels=age_distribution.index, 
        explode=(0.1,0,0), 
        autopct='%1.1f%%', 
        shadow=True,
        startangle=140
    )
    plt.axis('equal')
    plt.title('Distribution of Age Groups')
    plt.show()
    
    return df_cust

def analyze_gender_location_distribution(df_cust):
    """Analyze gender and location distribution"""
    print("\n" + "="*40)
    print("3. ANALYZE GENDER AND LOCATION DISTRIBUTION")
    print("="*40)
    
    customer_location_gender = df_cust.groupby(['location', 'gender']).size().unstack(fill_value=0)
    
    customer_location_gender.plot(kind='bar', stacked=True, figsize=(5, 4))
    plt.xlabel('Location')
    plt.ylabel('Count')
    plt.title('Customer Distribution by Location and Gender')
    plt.legend(title='Gender', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.show()

def clean_credit_score_data(df_cs):
    """Clean credit score data"""
    print("\n" + "="*50)
    print("EXPLORE CREDIT SCORE TABLE")
    print("="*50)
    
    print("Credit score data shape:", df_cs.shape)
    print("First few rows:")
    print(df_cs.head())
    
    # Step 1: Remove duplicates
    print("\n" + "="*40)
    print("DATA CLEANING STEP 1: REMOVE DUPLICATES")
    print("="*40)
    
    print(f"Unique customer IDs: {df_cs['cust_id'].nunique()}")
    duplicates = df_cs[df_cs.duplicated('cust_id', keep=False)]
    print(f"Duplicate records:")
    print(duplicates)
    
    df_cs_clean_1 = df_cs.drop_duplicates(subset='cust_id', keep="last")
    print(f"Shape after removing duplicates: {df_cs_clean_1.shape}")
    
    return df_cs_clean_1

def handle_credit_limit_nulls(df_cs_clean_1):
    """Handle null values in credit limit"""
    print("\n" + "="*40)
    print("DATA CLEANING STEP 2: HANDLE NULL VALUES")
    print("="*40)
    
    print("Null values:")
    print(df_cs_clean_1.isnull().sum())
    
    if df_cs_clean_1.credit_limit.isnull().sum() > 0:
        print("\nCredit limit null records:")
        print(df_cs_clean_1[df_cs_clean_1.credit_limit.isnull()])
        
        # Analyze relationship between credit score and credit limit
        print("\nCredit limit unique values:")
        print(df_cs_clean_1['credit_limit'].value_counts())
        
        # Visualize relationship
        plt.figure(figsize=(20, 5))
        plt.scatter(df_cs_clean_1['credit_limit'], df_cs_clean_1['credit_score'], c='blue', marker='o', label='Data Points')
        plt.title('Credit Score vs. Credit Limit')
        plt.xlabel('Credit Limit')
        plt.ylabel('Credit Score')
        plt.xticks(range(0, 90001, 5000))
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # Create credit score ranges
        bin_ranges = [300, 450, 500, 550, 600, 650, 700, 750, 800]
        bin_labels = [f'{start}-{end-1}' for start, end in zip(bin_ranges, bin_ranges[1:])]
        df_cs_clean_1['credit_score_range'] = pd.cut(df_cs_clean_1['credit_score'], bins=bin_ranges, labels=bin_labels, include_lowest=True, right=False)
        
        # Calculate mode per credit score range
        mode_df = df_cs_clean_1.groupby('credit_score_range')['credit_limit'].agg(lambda x: x.mode().iloc[0]).reset_index()
        print("\nMode credit limit per score range:")
        print(mode_df)
        
        # Merge and fill null values
        df_cs_clean_2 = pd.merge(df_cs_clean_1, mode_df, on='credit_score_range', suffixes=('', '_mode'))
        df_cs_clean_3 = df_cs_clean_2.copy()
        df_cs_clean_3['credit_limit'].fillna(df_cs_clean_3['credit_limit_mode'], inplace=True)
        
        print(f"\nNull values after treatment:")
        print(df_cs_clean_3.isnull().sum())
    
    return df_cs_clean_3

def handle_outstanding_debt_outliers(df_cs_clean_3):
    """Handle outliers in outstanding debt"""
    print("\n" + "="*40)
    print("DATA CLEANING STEP 3: HANDLE OUTLIERS: OUTSTANDING_DEBT")
    print("="*40)
    
    print("Credit score statistics:")
    print(df_cs_clean_3.describe())
    
    # Visualize outliers
    plt.figure(figsize=(5, 5))
    sns.boxplot(x=df_cs_clean_3['outstanding_debt'])
    plt.title('Box plot for outstanding debt')
    plt.show()
    
    # Find business rule outliers
    debt_outliers = df_cs_clean_3[df_cs_clean_3.outstanding_debt > df_cs_clean_3.credit_limit]
    print(f"\nOutstanding debt > credit limit:")
    print(debt_outliers)
    
    # Replace outliers with credit limit
    df_cs_clean_3.loc[df_cs_clean_3['outstanding_debt'] > df_cs_clean_3['credit_limit'], 'outstanding_debt'] = df_cs_clean_3['credit_limit']
    
    print(f"\nOutliers after treatment:")
    print(df_cs_clean_3[df_cs_clean_3.outstanding_debt > df_cs_clean_3.credit_limit])
    
    return df_cs_clean_3

def visualize_credit_correlations(df_cust, df_cs_clean_3):
    """Visualize correlations in merged data"""
    print("\n" + "="*40)
    print("DATA EXPLORATION: VISUALIZING CORRELATION")
    print("="*40)
    
    df_merged = df_cust.merge(df_cs_clean_3, on='cust_id', how='inner')
    
    numerical_cols = ['credit_score', 'credit_utilisation', 'outstanding_debt', 'credit_limit', 'annual_income', 'age']
    correlation_matrix = df_merged[numerical_cols].corr()
    
    print("Correlation matrix:")
    print(correlation_matrix)
    
    # Correlation heatmap
    plt.figure(figsize=(5, 3))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.8)
    plt.title('Correlation Plot')
    plt.show()
    
    # Scatter plot
    plt.figure(figsize=(5, 4))
    sns.scatterplot(x='annual_income', y='credit_score', data=df_merged, alpha=0.5)
    plt.title('Scatter Plot of Annual income vs credit score')
    plt.xlabel('Annual Income')
    plt.ylabel('Credit Score')
    plt.show()
    
    return df_merged

def clean_transactions_data(df_trans):
    """Clean transactions data"""
    print("\n" + "="*50)
    print("TRANSACTIONS TABLE")
    print("="*50)
    
    print("Transactions data shape:", df_trans.shape)
    print("First few rows:")
    print(df_trans.head(2))
    
    # Step 1: Handle null values in platform
    print("\n" + "="*40)
    print("DATA CLEANING STEP 1: HANDLE NULL VALUES: PLATFORM")
    print("="*40)
    
    print("Null values:")
    print(df_trans.isnull().sum())
    
    if df_trans.platform.isnull().sum() > 0:
        print("\nPlatform null records:")
        print(df_trans[df_trans.platform.isnull()])
        
        # Visualize platform distribution
        sns.countplot(y='product_category', hue='platform', data=df_trans)
        plt.show()
        
        # Fill with mode
        mode_platform = df_trans.platform.mode()[0]
        print(f"Mode platform: {mode_platform}")
        df_trans['platform'].fillna(mode_platform, inplace=True)
        
        print("Null values after treatment:")
        print(df_trans.isnull().sum())
    
    return df_trans

def handle_transaction_amount_outliers(df_trans):
    """Handle outliers in transaction amount"""
    print("\n" + "="*40)
    print("DATA CLEANING STEP 2: TREAT OUTLIERS: TRAN_AMOUNT")
    print("="*40)
    
    print("Transaction statistics:")
    print(df_trans.describe())
    
    # Handle zero transactions
    df_trans_zero = df_trans[df_trans.tran_amount == 0]
    print(f"\nZero amount transactions: {df_trans_zero.shape[0]}")
    
    if len(df_trans_zero) > 0:
        print("Zero transaction analysis:")
        print(df_trans_zero[['platform','product_category','payment_type']].value_counts())
        
        # Find median for replacement
        df_trans_1 = df_trans[(df_trans.platform=='Amazon') & 
                             (df_trans.product_category=="Electronics") & 
                             (df_trans.payment_type=="Credit Card")]
        
        median_to_replace = df_trans_1[df_trans_1.tran_amount > 0].tran_amount.median()
        print(f"Median for replacement: {median_to_replace}")
        
        df_trans['tran_amount'].replace(0, median_to_replace, inplace=True)
        
        print(f"Zero transactions after treatment: {len(df_trans[df_trans.tran_amount == 0])}")
    
    # Handle high-value outliers using IQR
    Q1, Q3 = df_trans['tran_amount'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    upper = Q3 + 2 * IQR
    
    print(f"\nOutlier threshold: {upper:.2f}")
    
    df_trans_outliers = df_trans[df_trans.tran_amount >= upper]
    df_trans_normal = df_trans[df_trans.tran_amount < upper]
    
    print(f"Outliers found: {len(df_trans_outliers)}")
    
    if len(df_trans_outliers) > 0:
        # Replace with category-wise mean
        tran_mean_per_category = df_trans_normal.groupby("product_category")["tran_amount"].mean()
        df_trans.loc[df_trans_outliers.index, 'tran_amount'] = df_trans_outliers['product_category'].map(tran_mean_per_category)
        
        print("Outliers replaced with category-wise mean")
    
    # Visualize final distribution
    sns.histplot(x='tran_amount', data=df_trans, bins=20, kde=True)
    plt.title('Transaction Amount Distribution (After Cleaning)')
    plt.show()
    
    return df_trans

def analyze_payment_patterns(df_merged, df_trans):
    """Analyze payment patterns and create visualizations"""
    print("\n" + "="*40)
    print("DATA VISUALIZATION: PAYMENT TYPE DISTRIBUTION")
    print("="*40)
    
    # Payment type distribution
    sns.countplot(x=df_trans.payment_type, stat='percent')
    plt.title('Payment Type Distribution')
    plt.show()
    
    # Merge all data
    df_merged_2 = df_merged.merge(df_trans, on='cust_id', how='inner')
    print(f"Final merged data shape: {df_merged_2.shape}")
    
    # Payment types by age group
    plt.figure(figsize=(5, 4))
    sns.countplot(x='age_group', hue='payment_type', data=df_merged_2, palette='Set3')
    plt.title('Distribution of Payment types across Age groups')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.legend(title='Payment Type', loc='upper right')
    plt.show()
    
    # Product category and platform by age group
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.countplot(x='age_group', hue="product_category", data=df_merged_2, ax=ax1)
    ax1.set_title("Product Category Count By Age Group")
    ax1.set_xlabel("Age Group")
    ax1.set_ylabel("Count")
    ax1.legend(title="Product Category", loc='upper right')
    
    sns.countplot(x='age_group', hue="platform", data=df_merged_2, ax=ax2)
    ax2.set_title("Platform Count By Age Group")
    ax2.set_xlabel("Age Group")
    ax2.set_ylabel("Count")
    ax2.legend(title="Platform", loc='upper right')
    
    plt.show()
    
    return df_merged_2

def analyze_transaction_amounts(df_merged_2):
    """Analyze average transaction amounts by different categories"""
    print("\n" + "="*40)
    print("DATA VISUALIZATION: AVERAGE TRANSACTION AMOUNT")
    print("="*40)
    
    cat_cols = ['payment_type', 'platform', 'product_category', 'marital_status', 'age_group']
    num_rows = 3
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))
    axes = axes.flatten()
    
    for i, cat_col in enumerate(cat_cols):
        avg_tran_amount_by_category = df_merged_2.groupby(cat_col)['tran_amount'].mean().reset_index()
        sorted_data = avg_tran_amount_by_category.sort_values(by='tran_amount', ascending=False)
        
        sns.barplot(x=cat_col, y='tran_amount', data=sorted_data, ci=None, ax=axes[i], palette='tab10')
        axes[i].set_title(f'Average transaction amount by {cat_col}')
        axes[i].set_xlabel(cat_col)
        axes[i].set_ylabel('Average transaction amount')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
    
    # Hide unused subplots
    for i in range(len(cat_cols), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

def analyze_age_group_metrics(df_merged):
    """Further analysis on age groups"""
    print("\n" + "="*40)
    print("FURTHER ANALYSIS ON AGE GROUP")
    print("="*40)
    
    # Calculate metrics by age group
    age_group_metrics = df_merged.groupby('age_group')[['annual_income', 'credit_limit', 'credit_score']].mean().reset_index()
    print("Age group metrics:")
    print(age_group_metrics)
    
    # Visualize metrics
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    sns.barplot(x='age_group', y='annual_income', data=age_group_metrics, palette='tab10', ax=ax1)
    ax1.set_title('Average Annual Income by Age Group')
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Average Annual Income')
    ax1.tick_params(axis='x', rotation=0)
    
    sns.barplot(x='age_group', y='credit_limit', data=age_group_metrics, palette='hls', ax=ax2)
    ax2.set_title('Average Credit Limit by Age Group')
    ax2.set_xlabel('Age Group')
    ax2.set_ylabel('Average Credit Limit')
    ax2.tick_params(axis='x', rotation=0)
    
    sns.barplot(x='age_group', y='credit_score', data=age_group_metrics, palette='viridis', ax=ax3)
    ax3.set_title('Average Credit Score by Age Group')
    ax3.set_xlabel('Age Group')
    ax3.set_ylabel('Average Credit Score')
    ax3.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    return age_group_metrics

def finalize_target_market():
    """Finalize target market recommendations"""
    print("\n" + "="*60)
    print("FINALIZE TARGET MARKET FOR TRIAL CREDIT CARD LAUNCH")
    print("="*60)
    
    print("TARGETING UNTAPPED MARKET")
    print("="*30)
    
    recommendations = [
        "1. People with age group of 18-25 account for ~26% of customer base",
        "2. Avg annual income of this group is less than 50k",
        "3. They don't have much credit history reflected in credit score and limit",
        "4. Usage of credit cards as payment type is relatively low vs other groups",
        "5. Top 3 shopping categories: Electronics, Fashion & Apparel, Beauty & Personal care"
    ]
    
    for rec in recommendations:
        print(rec)
    
    print("\nRECOMMENDATION:")
    print("Target the 18-25 age group for credit card launch as they represent")
    print("an untapped market with growth potential in credit card adoption.")

def main():
    """Main function to run the complete BREMARK Bank analysis"""
    print("BREMARK BANK CREDIT CARD LAUNCH ANALYSIS")
    print("="*70)
    
    # Data loading
    df_cust, df_cs, df_trans = load_data_from_csv()
    
    # Customer data analysis
    df_cust = explore_customers_data(df_cust)
    df_cust, occupation_wise_inc_median = handle_income_null_values(df_cust)
    df_cust = detect_and_treat_income_outliers(df_cust, occupation_wise_inc_median)
    # visualize_income_analysis(df_cust)
    
    df_cust = handle_age_outliers(df_cust)
    df_cust = create_age_groups(df_cust)
    analyze_gender_location_distribution(df_cust)
    
    # Credit score data analysis
    df_cs_clean_1 = clean_credit_score_data(df_cs)
    df_cs_clean_3 = handle_credit_limit_nulls(df_cs_clean_1)
    df_cs_clean_3 = handle_outstanding_debt_outliers(df_cs_clean_3)
    
    # Merge customer and credit data
    df_merged = visualize_credit_correlations(df_cust, df_cs_clean_3)
    
    # Transaction data analysis
    df_trans = clean_transactions_data(df_trans)
    df_trans = handle_transaction_amount_outliers(df_trans)
    
    # Final analysis
    df_merged_2 = analyze_payment_patterns(df_merged, df_trans)
    analyze_transaction_amounts(df_merged_2)
    age_group_metrics = analyze_age_group_metrics(df_merged)
    
    # Final recommendations
    finalize_target_market()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("Target Market: 18-25 age group for credit card launch")

if __name__ == "__main__":
    main()
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Define project paths
current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_directory, os.pardir))
sys.path.append(project_root)

# Load dataset
data_folder_path = os.path.join(project_root, 'Kaggle Dataset')
data_file_name = 'Kaggle_dataset.csv'
data = pd.read_csv(os.path.join(data_folder_path, data_file_name))

# Initialize parameters
start_date = datetime.strptime("2014-01-31", "%Y-%m-%d")
end_date = datetime.strptime("2024-10-31", "%Y-%m-%d")
monthly_churn_rate = 0.01            # 1% churn
new_customers_per_month = 3          # New customers per month
credit_limit_adjustment_rate = 0.03  # 3% of customers for credit limit adjustment

# Define function to generate monthly data with churn, new customers, and credit limit adjustments
def generate_monthly_data_with_new_customers(data, start_date, end_date, monthly_churn_rate, new_customers_per_month, credit_limit_adjustment_rate):
    # Create lists to store fabricated data
    all_data = []
    existing_customers = data.copy()  # Start with initial dataset
    max_customer_id = existing_customers['Customer ID'].max() + 1  # Track maximum Customer ID for new customers

    # Generate monthly data for each customer
    for month in pd.date_range(start=start_date, end=end_date, freq='M'):
        # Roll forward existing customers for this month
        month_data = existing_customers.copy()
        
        # Update Snapshot Month, Customer Age, and Month on Book
        month_data["Snapshot Month"] = month.strftime("%Y-%m-%d")
        month_data["Customer Age"] = month_data["Customer Age"] + (month.year - start_date.year)
        month_data["Month on Book"] = month_data["Month on Book"] + ((month.year - start_date.year) * 12 + month.month - start_date.month)
        
        # Simulate churn: Randomly select 1% of customers to churn
        churn_customers = month_data.sample(frac=monthly_churn_rate, random_state=1)
        month_data.loc[churn_customers.index, month_data.columns.difference(['Customer ID'])] = None  # Wipe out data except Customer ID

        # Adjust credit limits for 3% of customers to increase and 3% to decrease
        increase_customers = month_data.sample(frac=credit_limit_adjustment_rate, random_state=1).index
        decrease_customers = month_data.sample(frac=credit_limit_adjustment_rate, random_state=2).index

        # Increase credit limit by up to 10%
        month_data.loc[increase_customers, "Credit_Limit"] *= np.random.uniform(1.01, 1.10, size=len(increase_customers))
        
        # Decrease credit limit by up to 10%
        month_data.loc[decrease_customers, "Credit_Limit"] *= np.random.uniform(0.90, 0.99, size=len(decrease_customers))

        # Add new customers for this month
        new_customers = []
        for _ in range(new_customers_per_month):
            # Create a new customer entry with fabricated data
            new_customer = {
                'Snapshot Month': month.strftime("%Y-%m-%d"),
                'Customer ID': max_customer_id,
                'Customer Age': np.random.randint(18, 70),  # Random age between 18 and 70
                'Gender': np.random.choice(['M', 'F']),
                'Education_Level': np.random.choice(['Uneducated', 'High School', 'Graduate', 'Post-Graduate', 'Doctorate']),
                'Marital_Status': np.random.choice(['Single', 'Married', 'Unknown', 'Divorced']),
                'Income_Category': np.random.choice(['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']),
                'Month on Book': 1,  # New customers start with 1 month on book
                'Credit_Limit': np.random.randint(1000, 20000),  # Random credit limit between 1k and 20k
                'Revolving_Bal': np.random.randint(0, 5000),  # Random revolving balance between 0 and 5k
                'Utilization': np.round(np.random.uniform(0, 1), 3)  # Random utilization rate between 0 and 1
            }
            # Append new customer to the list for the current month
            new_customers.append(new_customer)
            # Increment customer ID for uniqueness
            max_customer_id += 1

        # Convert new customers for this month to DataFrame and add to month_data
        new_customers_df = pd.DataFrame(new_customers)
        month_data = pd.concat([month_data, new_customers_df], ignore_index=True)

        # Append fabricated data for this month to the main list
        all_data.append(month_data)
        
        # Update existing customers to include new customers for rolling forward in the next month
        existing_customers = pd.concat([existing_customers, new_customers_df], ignore_index=True)
    
    return pd.concat(all_data, ignore_index=True)

# Generate the fabricated data with rolling forward for new customers and credit limit adjustments
fabricated_data = generate_monthly_data_with_new_customers(data, start_date, end_date, monthly_churn_rate, new_customers_per_month, credit_limit_adjustment_rate)


# Adjust Revolving Balance based on customer profile and credit limit
def adjust_revolving_balance(data):
    # Define multipliers for Education_Level and Income_Category to influence Revolving_Bal
    education_multipliers = {
        'Uneducated': 1.0,
        'High School': 0.9,
        'Graduate': 0.7,
        'Post-Graduate': 0.6,
        'Doctorate': 0.5
    }

    income_multipliers = {
        'Less than $40K': 1.0,
        '$40K - $60K': 0.9,
        '$60K - $80K': 0.8,
        '$80K - $120K': 0.7,
        '$120K +': 0.6
    }

    # Iterate over each row in the dataset to calculate a Revolving_Bal
    def calculate_revolving_bal(row):
        # Base multiplier based on education and income
        education_multiplier = education_multipliers.get(row['Education_Level'], 1.0)
        income_multiplier = income_multipliers.get(row['Income_Category'], 1.0)

        # Age influence: older customers have a slightly reduced revolving balance
        age_factor = max(0.5, 1 - (row['Customer Age'] / 100))

        # Calculate the maximum possible revolving balance based on credit limit and multipliers
        max_possible_balance = row['Credit_Limit'] * education_multiplier * income_multiplier * age_factor

        # Ensure max_possible_balance is finite and non-negative, and handle NaN or excessive values
        if np.isnan(max_possible_balance) or max_possible_balance < 0 or row['Credit_Limit'] <= 0:
            return 0  # Fallback to 0 if invalid
        else:
            return np.random.uniform(0, min(max_possible_balance, row['Credit_Limit']))
    
    return data

# Apply revolving balance adjustments to the fabricated dataset
fabricated_data_with_adjusted_revolving_bal = adjust_revolving_balance(fabricated_data)

# Adjust Utilization based on Revolving Balance and Credit Limit
def adjust_utilization(data):
    # Calculate Utilization with a random factor, ensuring values are between 0 and 1
    def calculate_utilization(row):
        # Base utilization calculation
        base_util = row['Revolving_Bal'] / row['Credit_Limit'] if row['Credit_Limit'] > 0 else 0
        
        # Add some randomness, inversely proportional to the credit limit
        # Higher credit limits and lower balances get a nudge towards lower utilization
        random_factor = np.random.uniform(0.8, 1.0) if base_util < 0.5 else np.random.uniform(1.0, 1.2)
        utilization = base_util * random_factor

        # Ensure utilization stays within the 0-1 range
        return min(max(utilization, 0), 1)

    # Apply the calculation to the entire dataset
    data['Utilization'] = data.apply(calculate_utilization, axis=1)
    
    return data

# Apply utilization adjustments to the fabricated dataset
fabricated_data_with_adjusted_utilization = adjust_utilization(fabricated_data_with_adjusted_revolving_bal)

# Add two new variables based on utilization
def add_external_bank_credit_card_variables(data):
    # Define probabilities for values 0-9, skewed to make 4-9 less frequent
    value_probs = [0.3, 0.25, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
    
    # Generate the two variables with constraints and correlation to utilization
    def calculate_external_bank_util(row):
        # Define probabilities for values 0-9, skewed to make 4-9 less frequent
        value_probs = [0.3, 0.25, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]

        # Normalize probabilities if using a subset
        def normalize_probs(probs):
            total = sum(probs)
            return [p / total for p in probs] if total > 0 else probs

        # Base value for greater_than_90, influenced by utilization level
        if row['Utilization'] > 0.9:
            base_90 = np.random.choice(range(10), p=normalize_probs(value_probs))
        else:
            base_90 = np.random.choice(range(5), p=normalize_probs(value_probs[:5]))  # Lower values for lower utilization

        # Base value for greater_than_50, ensuring it's at least as large as base_90
        if row['Utilization'] > 0.5:
            base_50 = max(base_90, np.random.choice(range(10), p=normalize_probs(value_probs)))
        else:
            base_50 = max(base_90, np.random.choice(range(5), p=normalize_probs(value_probs[:5])))

        return base_90, base_50
    
    # Apply the calculation to each row
    data[['external_bank_credit_card_max_util_greater_than_90', 
          'external_bank_credit_card_max_util_greater_than_50']] = data.apply(
              lambda row: pd.Series(calculate_external_bank_util(row)), axis=1
          )
    
    return data

# Apply the function to add the two new variables
fabricated_data_with_external_bank_vars = add_external_bank_credit_card_variables(fabricated_data_with_adjusted_utilization)

# Add FICO score based on specified distribution and customer characteristics
def add_fico_scores(data):
    # Define the base FICO ranges with corresponding probabilities
    fico_ranges = {
        '400-679': (400, 679, 0.18),
        '680-719': (680, 719, 0.10),
        '720-759': (720, 759, 0.15),
        '760-799': (760, 799, 0.20),
        '800-850': (800, 850, 0.35)
    }

    # Assign a FICO range to each customer based on the probabilities
    fico_distribution = [fico_ranges[key] for key in fico_ranges]
    fico_probabilities = [r[2] for r in fico_distribution]
    
    # Normalize fico_probabilities to ensure they sum to 1
    total_prob = sum(fico_probabilities)
    fico_probabilities = [p / total_prob for p in fico_probabilities] if total_prob > 0 else fico_probabilities

    # Function to generate FICO score based on customer attributes
    def calculate_fico(row):
        # Determine base FICO range
        fico_min, fico_max = fico_distribution[np.random.choice(len(fico_distribution), p=fico_probabilities)][:2]
        base_fico = np.random.randint(fico_min, fico_max + 1)

        # Adjust FICO based on utilization, education, and income
        utilization_factor = 1 - row['Utilization']  # Lower utilization boosts FICO
        education_factor = {
            'Uneducated': 0.9, 'High School': 1.0, 'Graduate': 1.05,
            'Post-Graduate': 1.1, 'Doctorate': 1.15
        }.get(row['Education_Level'], 1.0)
        income_factor = {
            'Less than $40K': 0.9, '$40K - $60K': 1.0,
            '$60K - $80K': 1.05, '$80K - $120K': 1.1, '$120K +': 1.15
        }.get(row['Income_Category'], 1.0)

        # Apply adjustments and cap FICO score between 400 and 850
        adjusted_fico = min(max(int(base_fico * utilization_factor * education_factor * income_factor), 400), 850)

        # Introduce missing values based on a random draw
        if np.random.rand() < 0.02:  # 2% missing values
            return None
        return adjusted_fico

    # Apply the function to each row
    data['FICO'] = data.apply(calculate_fico, axis=1)
    
    return data

# Apply the function to add FICO scores to the dataset
fabricated_data_with_fico = add_fico_scores(fabricated_data_with_external_bank_vars)

# Add Total Debt based on customer profile and realistic debt categories
def add_total_debt(data):
    # Function to calculate total debt for each customer
    def calculate_total_debt(row):
        # Base Credit Card Debt (from Revolving Balance)
        credit_card_debt = row['Revolving_Bal']
        
        # Student Loan: Primarily for younger, educated customers
        if row['Customer Age'] < 40 and row['Education_Level'] in ['Graduate', 'Post-Graduate', 'Doctorate']:
            student_loan = np.random.uniform(5000, 30000)  # Range for student loans
        else:
            student_loan = 0
        
        # Car Loan: Common for middle-aged customers (ages 25–55)
        if 25 <= row['Customer Age'] <= 55:
            car_loan = np.random.uniform(5000, 40000)  # Range for car loans
        else:
            car_loan = 0
        
        # Mortgage: More likely for customers over 30 with higher incomes
        income_factor = {
            'Less than $40K': 0,
            '$40K - $60K': 0.3,
            '$60K - $80K': 0.6,
            '$80K - $120K': 0.8,
            '$120K +': 1.0
        }.get(row['Income_Category'], 0.5)
        
        if row['Customer Age'] > 30 and income_factor > 0.5:
            mortgage = np.random.uniform(50000, 500000) * income_factor  # Range adjusted by income factor
        else:
            mortgage = 0
        
        # Calculate total debt as sum of all components
        total_debt = credit_card_debt + student_loan + car_loan + mortgage
        
        return total_debt
    
    # Apply the function to each row in the dataset
    data['Total_Debt'] = data.apply(calculate_total_debt, axis=1)
    
    return data

# Apply the function to add Total Debt to the dataset
fabricated_data_with_total_debt = add_total_debt(fabricated_data_with_fico)

# Add Debt-to-Income Ratio based on Total Debt and representative income within income levels
def add_dti_ratio(data):
    # Define representative income values for each Income Category, with slight random variation
    income_values = {
        'Less than $40K': 30000,
        '$40K - $60K': 50000,
        '$60K - $80K': 70000,
        '$80K - $120K': 100000,
        '$120K +': 150000
    }
    
    # Function to calculate DTI based on Total Debt and representative income
    def calculate_dti(row):
        # Get base income from income category, apply slight random variation
        base_income = income_values.get(row['Income_Category'], 50000)  # Default to 50K if category is missing
        income_with_variation = base_income * np.random.uniform(0.9, 1.1)  # Apply 10% variation
        
        # Calculate DTI ratio, ensuring it stays within 0-1 range
        if income_with_variation > 0:
            dti_ratio = min(row['Total_Debt'] / income_with_variation, 1)
        else:
            dti_ratio = 0  # Set DTI to 0 if income is 0 or undefined
            
        return dti_ratio
    
    # Apply the function to each row in the dataset
    data['Debt_to_Income_Ratio'] = data.apply(calculate_dti, axis=1)
    
    return data

# Apply the function to add Debt-to-Income Ratio to the dataset
fabricated_data_with_dti = add_dti_ratio(fabricated_data_with_total_debt)

# Add Credit Inquiries based on customer profile
def add_credit_inquiries(data):
    # Function to calculate credit inquiries based on influencing factors
    def calculate_credit_inquiries(row):
        # Define base inquiry count with a slight random factor
        base_inquiries = np.random.randint(0, 4)  # Starting point, between 0 and 3 inquiries
        
        # Adjust based on FICO score: lower FICO tends to have more inquiries
        if row['FICO'] is not None:
            if row['FICO'] < 600:
                base_inquiries += np.random.randint(2, 5)  # Add 2-4 more inquiries
            elif 600 <= row['FICO'] < 700:
                base_inquiries += np.random.randint(1, 3)  # Add 1-2 more inquiries
            elif row['FICO'] > 750:
                base_inquiries = max(0, base_inquiries - np.random.randint(1, 2))  # Decrease for high FICO
        
        # Adjust based on utilization: higher utilization could lead to more inquiries
        if row['Utilization'] > 0.8:
            base_inquiries += np.random.randint(1, 4)
        
        # Adjust based on age: younger customers tend to have more inquiries
        if row['Customer Age'] < 30:
            base_inquiries += np.random.randint(1, 3)
        elif row['Customer Age'] > 50:
            base_inquiries = max(0, base_inquiries - np.random.randint(1, 2))  # Fewer inquiries for older customers

        # Ensure inquiries stay within a realistic range of 0 to 12
        return min(max(base_inquiries, 0), 12)
    
    # Apply the function to each row in the dataset
    data['Credit_Inquiries'] = data.apply(calculate_credit_inquiries, axis=1)
    
    return data

# Apply the function to add Credit Inquiries to the dataset
fabricated_data_with_inquiries = add_credit_inquiries(fabricated_data_with_dti)

def add_delinquency(data):
    # Step 1: Assign random delinquency levels (0 or 1) with 1.5% chance of 1
    delinquency_probs = [0.985, 0.015]  # 98.5% chance of 0, 1.5% chance of 1
    data['Delinquency'] = np.random.choice([0, 1], size=len(data), p=delinquency_probs)

    # Sort data by Customer ID and Snapshot Month to ensure proper order
    data = data.sort_values(['Customer ID', 'Snapshot Month']).reset_index(drop=True)

    # Step 2: Apply weighted progression to customers with Delinquency=1
    for customer_id, customer_data in data.groupby('Customer ID'):
        # Get indices for the current customer's data
        customer_indices = customer_data.index
        
        # Find indices where delinquency is 1
        delinquency_1_indices = customer_data[customer_data['Delinquency'] == 1].index
        
        # Progression based on high-risk factors
        for start_index in delinquency_1_indices:
            row = data.loc[start_index]
            
            # Check high-risk conditions to determine progression probability
            high_risk = (
                row['Utilization'] > 0.8 or
                row['Education_Level'] in ['Uneducated', 'High School'] or
                row['Income_Category'] in ['Less than $40K', '$40K - $60K'] or
                row['Revolving_Bal'] > 3000
            )
            
            # Roll forward 60% for high-risk customers, otherwise 30%
            roll_forward_prob = 0.6 if high_risk else 0.3
            if np.random.rand() < roll_forward_prob:
                # Increase delinquency levels over the next 6 months, capping at 7
                for offset in range(1, 7):
                    next_index = start_index + offset
                    if next_index in customer_indices:
                        data.loc[next_index, 'Delinquency'] = min(1 + offset, 7)
                
                # Keep delinquency at 7 for the next 11 months
                for offset in range(7, 18):
                    hold_index = start_index + offset
                    if hold_index in customer_indices:
                        data.loc[hold_index, 'Delinquency'] = 7
                
                # Attrition: Wipe data but retain Customer ID after 11 months at delinquency 7
                attrition_start_index = start_index + 18
                for attrition_index in range(attrition_start_index, len(data)):
                    if attrition_index in customer_indices:
                        data.loc[attrition_index, data.columns.difference(['Customer ID'])] = None

    return data

# Apply the function to add delinquency progression with weighted criteria
fabricated_data_with_delinquency = add_delinquency(fabricated_data_with_inquiries)

# Add Monthly Interest Revenue based on a monthly interest rate derived from 22.75% annual rate
def add_monthly_interest_revenue(data):
    # Calculate monthly interest revenue as 1.8958% of the revolving balance
    data['Monthly_Interest_Revenue'] = data['Revolving_Bal'] * 0.018958
    return data

# Apply the function to add Monthly Interest Revenue to the dataset
fabricated_data_with_monthly_interest_revenue = add_monthly_interest_revenue(fabricated_data_with_delinquency)

def add_late_fee_revenue(data):
    # Assign $30 for any customer with delinquency > 0; otherwise, $0
    data['Late_Fee_Revenue'] = data['Delinquency'].apply(lambda x: 30 if x > 0 else 0)
    return data

# Apply the function to add Late Fee Revenue to the dataset
fabricated_data_with_late_fee_revenue = add_late_fee_revenue(fabricated_data_with_monthly_interest_revenue)

# Add Annual Fee based on Month on Book
def add_annual_fee(data):
    # Assign $100 for rows where Month on Book is a multiple of 12; otherwise, $0
    data['Annual_Fee'] = data['Month on Book'].apply(lambda x: 100 if x % 12 == 0 else 0)
    return data

# Apply the function to add Annual Fee to the dataset
fabricated_data_with_annual_fee = add_annual_fee(fabricated_data_with_late_fee_revenue)

def add_ecl(data):
    # Define the ECL rates for each delinquency level from 0 to 7
    ecl_rates = [0.05, 0.40, 0.60, 0.63, 0.81, 0.81, 0.81, 0.85]

    # Calculate ECL based on Delinquency and Revolving Balance, handling NaN values in Delinquency
    data['ECL'] = data.apply(lambda row: row['Revolving_Bal'] * ecl_rates[int(row['Delinquency'])]
                             if pd.notna(row['Delinquency']) else 0, axis=1)
    
    return data

# Apply the function to add ECL to the dataset
fabricated_data_with_ecl = add_ecl(fabricated_data_with_late_fee_revenue)

def add_ecl_mom_charge(data):
    # Ensure data is sorted by Customer ID and Snapshot Month for proper calculation
    data = data.sort_values(['Customer ID', 'Snapshot Month']).reset_index(drop=True)

    # Initialize the ECL MoM Charge column with NaN
    data['ECL MoM Charge'] = np.nan

    # Calculate ECL MoM Charge for each customer
    for customer_id, customer_data in data.groupby('Customer ID'):
        previous_ecl = None  # Track the ECL of the previous month for each customer

        for index in customer_data.index:
            current_ecl = data.loc[index, 'ECL']
            if pd.isna(previous_ecl):  # If no previous ECL (first month or new data after attrition)
                data.loc[index, 'ECL MoM Charge'] = current_ecl
            else:
                data.loc[index, 'ECL MoM Charge'] = current_ecl - previous_ecl
            
            # Update previous ECL for the next iteration
            previous_ecl = current_ecl

    return data

# Apply the function to add ECL MoM Charge to the dataset
fabricated_data_with_ecl_mom_charge = add_ecl_mom_charge(fabricated_data_with_ecl)

print(fabricated_data_with_ecl_mom_charge.shape)
print(fabricated_data_with_ecl_mom_charge.head())
# data_cleaning.py - Production-grade ETL
import mysql.connector
import pandas as pd
from sklearn.preprocessing import PowerTransformer

def validate_data(df):
    """ Validates and cleans the input DataFrame. """
    required_columns = ['CustomerID', 'Tenure', 'MonthlySpend', 'ContractType', 
                        'SubscriptionPlan', 'LoginFrequency', 'SupportTickets', 'ChurnStatus']
    
    # Ensure all required columns exist
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {set(required_columns) - set(df.columns)}")

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_columns)
    
    return df

def transform_features(df):
    """ Transforms skewed features and computes additional metrics. """
    pt = PowerTransformer()
    df[['MonthlySpend', 'Tenure']] = pt.fit_transform(df[['MonthlySpend', 'Tenure']])

    # Compute additional features
    df['SpendChangeRate'] = df.groupby('CustomerID')['MonthlySpend'].pct_change().fillna(0)
    df['TicketResolutionRate'] = df['SupportTickets'] / (df['LoginFrequency'] + 1)

    # Fill NaN values with 0 to avoid MySQL errors
    df.fillna(0, inplace=True)
    
    return df

def load_to_mysql(df):
    """ Loads transformed data into MySQL database. """
    conn = None  
    try:
        conn = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='dhawal@sql'  # Ensure this password is correct
        )
        
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS saas_analysis;")
        cursor.execute("USE saas_analysis;")  # Switch to the database
        
        # Create table with correct schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                CustomerID VARCHAR(50) PRIMARY KEY,
                Tenure FLOAT,
                MonthlySpend FLOAT,
                ContractType VARCHAR(20),
                SubscriptionPlan VARCHAR(20),
                LoginFrequency INT,
                SupportTickets INT,
                ChurnStatus INT,
                SpendChangeRate FLOAT,
                TicketResolutionRate FLOAT
            )
        ''')

        insert_query = '''
            INSERT INTO customers 
            (CustomerID, Tenure, MonthlySpend, ContractType, SubscriptionPlan, 
            LoginFrequency, SupportTickets, ChurnStatus, SpendChangeRate, TicketResolutionRate)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            Tenure=VALUES(Tenure),
            MonthlySpend=VALUES(MonthlySpend),
            ContractType=VALUES(ContractType),
            SubscriptionPlan=VALUES(SubscriptionPlan),
            LoginFrequency=VALUES(LoginFrequency),
            SupportTickets=VALUES(SupportTickets),
            ChurnStatus=VALUES(ChurnStatus),
            SpendChangeRate=VALUES(SpendChangeRate),
            TicketResolutionRate=VALUES(TicketResolutionRate)
        '''

        # Ensure correct column ordering for insertion
        for _, row in df.iterrows():
            row_tuple = (
                row['CustomerID'], row['Tenure'], row['MonthlySpend'], row['ContractType'], row['SubscriptionPlan'],
                row['LoginFrequency'], row['SupportTickets'], row['ChurnStatus'], row['SpendChangeRate'], row['TicketResolutionRate']
            )
            cursor.execute(insert_query, row_tuple)

        conn.commit()
        print(f"Successfully loaded {len(df)} records")

    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")

    finally:
        if conn is not None and conn.is_connected():
            cursor.close()
            conn.close()
            print("MySQL connection closed")

if __name__ == "__main__":
    # Load dataset
    raw_data = pd.read_csv("saas_dataset.csv")
    
    # Clean and transform the data
    cleaned_data = validate_data(raw_data)
    transformed_data = transform_features(cleaned_data)
    
    # Load data into MySQL
    load_to_mysql(transformed_data)

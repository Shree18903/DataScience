import pandas as pd
import json
import numpy as np

# Load the dataset
file_path = 'TRADES_CopyTr_90D_ROI.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Handle missing values in Trade_History
data = data.dropna(subset=['Trade_History'])

# Debugging: Inspect raw Trade_History entries
print("Raw Trade_History entries (first 5 rows):")
print(data['Trade_History'].head(5).tolist())

# Parse JSON-like strings in Trade_History
def parse_trade_history(entry):
    try:
        # Attempt to parse JSON-like strings
        return json.loads(entry.replace("'", '"'))  # Replace single quotes with double quotes for valid JSON
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {entry}\nError: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {entry}\nError: {e}")
        return None

data['Trade_History'] = data['Trade_History'].apply(parse_trade_history)

# Drop rows where Trade_History could not be parsed
data = data.dropna(subset=['Trade_History'])

# Expand Trade_History into individual rows
data_expanded = data.explode('Trade_History').reset_index(drop=True)

# Convert nested JSON into columns
if 'Trade_History' in data_expanded.columns:
    try:
        trade_details = pd.concat(
            [data_expanded.drop(['Trade_History'], axis=1),
             data_expanded['Trade_History'].apply(pd.Series)], axis=1)
    except Exception as e:
        print(f"Error expanding Trade_History into columns: {e}")
        exit()
else:
    print("Error: Trade_History column is not properly parsed. Please check the data format.")
    exit()

# Debugging: Check available columns in trade_details
print("Available columns in trade_details:", trade_details.columns.tolist())

# Check for required fields
required_fields = ['quantity', 'realizedProfit']
missing_fields = [field for field in required_fields if field not in trade_details.columns]

if missing_fields:
    print(f"Error: Missing required fields: {missing_fields}")
    print("Sample of parsed trade_details:")
    print(trade_details.head(10))  # Show 10 rows for better debugging
    exit()

# Ensure numeric columns are properly cast
for col in required_fields:
    trade_details[col] = pd.to_numeric(trade_details[col], errors='coerce')

# Calculate metrics for each account
def calculate_metrics(group):
    total_investment = group['quantity'].sum()
    total_profit = group['realizedProfit'].sum()
    positions = len(group)
    win_positions = (group['realizedProfit'] > 0).sum()
    win_rate = win_positions / positions if positions > 0 else 0

    # ROI
    roi = (total_profit / total_investment) * 100 if total_investment > 0 else 0

    # Sharpe Ratio (Assume risk-free rate = 0 for simplicity)
    returns = group['realizedProfit']
    sharpe_ratio = (returns.mean() / returns.std()) if returns.std() > 0 else 0

    # Maximum Drawdown
    cumulative_profit = returns.cumsum()
    mdd = (cumulative_profit - cumulative_profit.cummax()).min()

    return pd.Series({
        'ROI': roi,
        'PnL': total_profit,
        'Sharpe_Ratio': sharpe_ratio,
        'MDD': mdd,
        'Win_Rate': win_rate,
        'Win_Positions': win_positions,
        'Total_Positions': positions
    })

metrics = trade_details.groupby('Port_IDs', group_keys=False).apply(calculate_metrics).reset_index()

# Rank accounts based on ROI
metrics['Rank'] = metrics['ROI'].rank(ascending=False, method='min')

# Select top 20 accounts
top_20_accounts = metrics.nsmallest(20, 'Rank')

# Save results to CSV
metrics.to_csv('calculated_metrics.csv', index=False)
top_20_accounts.to_csv('top_20_accounts.csv', index=False)

# Print top 20 accounts
print("Top 20 accounts based on ROI:")
print(top_20_accounts)

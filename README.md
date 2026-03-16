import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Create DataFrame
data = {"salary": [25000, 30000, 28000, 35000]}
df = pd.DataFrame(data)

# Initialize scaler
scaler = MinMaxScaler()

# Apply normalization (fit_transform expects 2D input)
df["Normalized"] = scaler.fit_transform(df[["salary"]])

# Print result
print(df)

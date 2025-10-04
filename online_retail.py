# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("online_retail.csv", encoding="ISO-8859-1")

# -----------------------------
# Step 1: Data Cleaning
# -----------------------------
df = df.drop_duplicates()

df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')

df = df.dropna(subset=['CustomerID', 'Quantity', 'UnitPrice', 'InvoiceDate'])

df['CustomerID'] = df['CustomerID'].astype(int)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# -----------------------------
# Step 2: Define Reference Date
# -----------------------------
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# -----------------------------
# Step 3: Calculate RFM Metrics
# -----------------------------
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,   # Recency
    'InvoiceNo': 'nunique',                                     # Frequency
    'TotalPrice': 'sum'                                         # Monetary
}).reset_index()

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
}, inplace=True)

# -----------------------------
# Step 4: Assign RFM Scores
# -----------------------------
rfm['R_score'] = pd.qcut(rfm['Recency'].rank(method="first"), 5, labels=[5,4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1,2,3,4,5])
rfm['M_score'] = pd.qcut(rfm['Monetary'].rank(method="first"), 5, labels=[1,2,3,4,5])

rfm['RFM_Segment'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
rfm['RFM_Score'] = rfm[['R_score','F_score','M_score']].astype(int).sum(axis=1)

# -----------------------------
# Step 5: Customer Segmentation
# -----------------------------
def segment_customer(row):
    r, f, m = int(row['R_score']), int(row['F_score']), int(row['M_score'])
    if r >= 4 and f >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3:
        return 'Loyal Customers'
    elif r >= 3 and f <= 2:
        return 'Potential Loyalist'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# -----------------------------
# Step 6: Visualization
# -----------------------------

# (1) Existing: Segment distribution
plt.figure(figsize=(10,6))
sns.countplot(x="Segment", data=rfm, order=rfm['Segment'].value_counts().index, palette="viridis")
plt.title("Customer Segments Distribution (RFM)")
plt.xlabel("Customer Segment")
plt.ylabel("Count of Customers")
plt.show()

# (2) Existing: RFM heatmap
rfm_pivot = rfm.pivot_table(index='F_score', columns='R_score', values='Monetary', aggfunc='mean')
plt.figure(figsize=(8,6))
sns.heatmap(rfm_pivot, cmap="YlGnBu", annot=True, fmt=".1f")
plt.title("Heatmap of RFM Segments (Monetary Value)")
plt.show()

# (3) NEW: Distribution of Recency, Frequency, Monetary
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(rfm['Recency'], bins=30, kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Distribution of Recency (days)")

sns.histplot(rfm['Frequency'], bins=30, kde=True, ax=axes[1], color="orange")
axes[1].set_title("Distribution of Frequency (transactions)")

sns.histplot(rfm['Monetary'], bins=30, kde=True, ax=axes[2], color="green")
axes[2].set_title("Distribution of Monetary Value (£)")
plt.show()

# (4) NEW: Boxplots to detect outliers
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(y=rfm['Recency'], ax=axes[0], color="skyblue")
axes[0].set_title("Recency Outliers")

sns.boxplot(y=rfm['Frequency'], ax=axes[1], color="orange")
axes[1].set_title("Frequency Outliers")

sns.boxplot(y=rfm['Monetary'], ax=axes[2], color="green")
axes[2].set_title("Monetary Outliers")
plt.show()

# (5) NEW: Scatter plot (Frequency vs Monetary)
plt.figure(figsize=(8,6))
sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Segment', palette="viridis", alpha=0.7)
plt.title("Frequency vs Monetary by Segment")
plt.xlabel("Frequency (transactions)")
plt.ylabel("Monetary Value (£)")
plt.legend(title="Segment")
plt.show()

# (6) NEW: Average RFM values per Segment
rfm_segment_means = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean().reset_index()
rfm_segment_means.plot(x='Segment', kind='bar', figsize=(10,6))
plt.title("Average R, F, M by Customer Segment")
plt.ylabel("Average Value")
plt.show()

# -----------------------------
# Step 7: Save Results
# -----------------------------
rfm.to_csv("RFM_Customer_Segments.csv", index=False)
print("✅ RFM segmentation completed and results saved to 'RFM_Customer_Segments.csv'")

###############################################################################################
# Task 1: Data Preparation
###############################################################################################
# Step 1: Read the 2010-2011 sheet from the Online Retail II dataset.
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T

#                 count          mean          std       min       25%       50%       75%      max
# Quantity     541910.0      9.552234   218.080957 -80995.00      1.00      3.00     10.00  80995.0
# Price        541910.0      4.611138    96.759765 -11062.06      1.25      2.08      4.13  38970.0
# Customer ID  406830.0  15287.684160  1713.603074  12346.00  13953.00  15152.00  16791.00  18287.0

df.isnull().sum()
# Invoice             0
# StockCode           0
# Description      1454
# Quantity            0
# InvoiceDate         0
# Price               0
# Customer ID    135080
# Country             0
# dtype: int64

# Step 2: Drop observation units with StockCode equal to POST.
# (POST does not represent a product price, but an additional charge added to each invoice.)
post_index = df.loc[df["StockCode"] == "POST"].index
df.drop(index=post_index, inplace=True)
# Step 3: Drop observation units with missing values.
df.dropna(inplace=True)
# Step 4: Remove values in the dataset where Invoice contains C. (C indicates a cancelled invoice.)
df = df[~df["Invoice"].str.contains("C", na=False)]
# Step 5: Filter observation units where Price is less than zero.
df = df[df["Price"] > 0]
# Step 6: Examine and possibly suppress outliers in the Price and Quantity variables.
def outlier_thresholds(dataframe, variable):
    # Find outliers in variables.
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    # During the code execution, check for the presence of outliers and perform a task if they exist or direct to perform a task if they do not.
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False
for col in ["Price", "Quantity"]:
    print(col, check_outlier(df, col))
# Price True
# Quantity True

def replace_with_thresholds(dataframe, variable):
    # Suppress outliers with lower and upper limits.
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)

###############################################################################################
# Task 2: Creating Association Rules through German Customers
###############################################################################################
# Step 1: Define the create_invoice_product_df function to build a pivot table for invoice products as shown below.
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

create_invoice_product_df(df, id=True).iloc[0:5, 0:5]
# StockCode  10002  10080  10120  10125  10133
# Invoice
# 536365         0      0      0      0      0
# 536366         0      0      0      0      0
# 536367         0      0      0      0      0
# 536368         0      0      0      0      0
# 536369         0      0      0      0      0

# Step 2: Define the create_rules function to generate rules and find the rules for German customers.

def create_rules(dataframe, id=True, country="Germany"):
    dataframe = dataframe[dataframe["Country"] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

rules = create_rules(df)

#       antecedents                          consequents  antecedent support  consequent support   support  confidence       lift  leverage  conviction
# 0         (16237)                              (22326)            0.011136            0.249443  0.011136    1.000000   4.008929  0.008358         inf
# 1         (22326)                              (16237)            0.249443            0.011136  0.011136    0.044643   4.008929  0.008358    1.035073
# 2         (20674)                              (20675)            0.022272            0.033408  0.013363    0.600000  17.960000  0.012619    2.416481
# 3         (20675)                              (20674)            0.033408            0.022272  0.013363    0.400000  17.960000  0.012619    1.629547
# 4         (20674)                              (20676)            0.022272            0.037862  0.011136    0.500000  13.205882  0.010293    1.924276
#            ...                                  ...                 ...                 ...       ...         ...        ...       ...         ...
# 18365     (22629)  (22467, 22326, 22423, 21915, 22077)            0.104677            0.011136  0.011136    0.106383   9.553191  0.009970    1.106586
# 18366     (22326)  (22467, 22629, 22423, 21915, 22077)            0.249443            0.011136  0.011136    0.044643   4.008929  0.008358    1.035073
# 18367     (22423)  (22467, 22629, 22326, 21915, 22077)            0.140312            0.011136  0.011136    0.079365   7.126984  0.009573    1.074111
# 18368     (21915)  (22467, 22629, 22326, 22423, 22077)            0.046771            0.011136  0.011136    0.238095  21.380952  0.010615    1.297884
# 18369     (22077)  (22467, 22629, 22326, 22423, 21915)            0.104677            0.011136  0.011136    0.106383   9.553191  0.009970    1.106586
# [18370 rows x 9 columns]
###############################################################################################
# Task 3: Providing Product Recommendations to Users with Given Product IDs in Their Baskets
###############################################################################################
# Step 1: Use the check_id function to find the names of the given products.
def check_in(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)
check_in(df, 21987) # ['PACK OF 6 SKULL PAPER CUPS']
check_in(df, 23235) # ['STORAGE TIN VINTAGE LEAF']
check_in(df, 22747) # ["POPPY'S PLAYHOUSE BATHROOM"]

# Step 2: Use the arl_recommender function to provide product recommendations for 3 users.
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]

user_1 = arl_recommender(rules, 21987, 1) # [21989]
user_2 = arl_recommender(rules, 23235, 1) # [23244]
user_3 = arl_recommender(rules, 22747, 1) # [22746]
# Step 3: Check the names of the recommended products.

for_user_1 = check_in(df, 21989) # ['PACK OF 20 SKULL PAPER NAPKINS']
for_user_2 = check_in(df, 23244) # ['ROUND STORAGE TIN VINTAGE LEAF']
for_user_3 = check_in(df, 22746) # ["POPPY'S PLAYHOUSE LIVINGROOM "]











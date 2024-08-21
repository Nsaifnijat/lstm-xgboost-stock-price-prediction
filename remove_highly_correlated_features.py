import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('rawfeatures.csv',index_col='time')

df = df.drop(columns=['open','high','low'])

def plot_correlation(df):
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    plt.figure(figsize=(25, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.show()


#plot_correlation(df)

def remove_highly_correlated_pairs(df,threshold):
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
   
    # Find features with correlation greater than 0.9
    threshold = 0.9
    to_drop = set()

    # Iterate over the columns of the upper triangle
    for col in upper.columns:
        for row in upper.index:
            if abs(upper.at[row, col]) > threshold:  # Check if correlation exceeds the threshold
                # Add one of the highly correlated features to the drop list
                to_drop.add(col)  # This keeps 'row' and drops 'col', could be adjusted based on criteria

    # Drop the features from the DataFrame
    df_reduced = df.drop(columns=to_drop)
    # Print the features that will be dropped
    print(f"Features to drop: {to_drop}")
    return df_reduced

df_eliminated = remove_highly_correlated_pairs(df, 0.9)





#or 
def remove_features_based_on_variance_threshold(df):
    from sklearn.feature_selection import VarianceThreshold, RFE
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    selector = VarianceThreshold(threshold=0.1)
    df_reduced_var = selector.fit_transform(df)
    # Convert back to DataFrame
    df_reduced_var = pd.DataFrame(df_reduced_var, columns=df.columns[selector.get_support(indices=True)], index=df.index)

    return df_reduced_var

#df_eliminated = remove_features_based_on_variance_threshold(df)

"""
Recursive Feature Elimination (RFE): 
Use RFE to select features by recursively considering smaller sets of features."""
def recursive_elimination(df):
    import pandas as pd
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    # Define the model and RFE
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=10)  # Adjust n_features_to_select based on your requirement

    # Fit RFE
    fit = rfe.fit(df.drop(columns=['target']), df['target'])

    # Select the features that were supported by RFE
    df_reduced_rfe = df[df.columns[fit.support_]]

    # Optionally, include the target column in the reduced DataFrame
    df_reduced_rfe['target'] = df['target']

    # Save the reduced DataFrame to a CSV file
    df_reduced_rfe.to_csv('reduced_data.csv', index=False)

    print("Features selected by RFE:")
    print(df_reduced_rfe.columns)


#df_eliminated = recursive_elimination(df)

print(df_eliminated[:5])
print(df_eliminated.columns)
# Save to CSV
df_eliminated.to_csv('uncorrelatedFeatures.csv')
print(df_eliminated.columns)

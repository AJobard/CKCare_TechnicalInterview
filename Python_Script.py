# -*- coding: utf-8 -*-

"""# Import libraries and load dataframe."""


# Import useful PYTHON libraries.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import CSV files extracted from the instruction HTML page - Using local interface (For sensitive data)

file_path_Clinical_Data = './Clinical_Data.csv'
file_path_BiologicalSample_Data = './BiologicalSample_Data.csv'

df1 = pd.read_csv(file_path_Clinical_Data)
df2 = pd.read_csv(file_path_BiologicalSample_Data)

df1.shape
# df2.shape

# Displaying dataframe.

df1.head()

# Displaying dataframe.

df2.head()

# General structure of dataframe.

print(df1.info())
print(df2.info())
# print(df3.info())

"""# Quality check"""

# Check about duplicata for patient (ID)

duplicates_df1 = df1['ID'].duplicated().sum()
duplicates_df2 = df2['ID'].duplicated().sum()

print(f"Duplicata for ID column with df1 : {duplicates_df1}")
print(f"Duplicata for ID column with df2 : {duplicates_df2}")

# Histogram for non-null values on df1.

non_null_counts = df1.notnull().sum()
non_null_counts_sorted = non_null_counts.sort_values(ascending=False)
plt.figure(figsize=(80, 6))
non_null_counts_sorted.plot(kind='bar', color='skyblue')
plt.title("Non null values per columns into df1 ")
plt.xlabel("Column")
plt.ylabel("Number of non null values")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Histogram for non null values on df2.

def count_valid_values(series):
    return series[~series.isnull() & (series != '') & (series != ' ')].count()

# Application on df2.

valid_counts = df2.apply(count_valid_values)
valid_counts = valid_counts.reindex(df2.columns, fill_value=0)
valid_counts_sorted = valid_counts.sort_values(ascending=False)
plt.figure(figsize=(20, 6))
valid_counts_sorted.plot(kind='bar', color='skyblue')
plt.title("Non-null and non Nan values per columns")
plt.xlabel("Columns")
plt.ylabel("Number of valid values")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Boxplot for numerical column on df1 - Quick visualisation of outliers.

numeric_cols = df1.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(20, 15))
for i, col in enumerate(numeric_cols):
    plt.subplot(6, 3, i+1)
    plt.boxplot(df1[col].dropna())
    plt.title(col)
    plt.xticks([])
plt.tight_layout()
plt.show()

# Boxplot for numerical column on df1 - Labelled boxplot.

numeric_cols = df1.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(40, 35))
for i, col in enumerate(numeric_cols):
    plt.subplot(6, 3, i+1)
    data = df1[col].dropna()
    box = plt.boxplot(data, patch_artist=True)

    # Key values
    q1 = np.percentile(data, 25)
    median = np.median(data)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr
    outliers = data[(data < lower_whisker) | (data > upper_whisker)]

    # Add labels
    plt.title(col)
    plt.xticks([])
    plt.annotate(f'Median: {median:.2f}', xy=(1, median), xytext=(1.1, median), arrowprops=dict(arrowstyle='->'))
    plt.annotate(f'Q1: {q1:.2f}', xy=(1, q1), xytext=(1.1, q1), arrowprops=dict(arrowstyle='->'))
    plt.annotate(f'Q3: {q3:.2f}', xy=(1, q3), xytext=(1.1, q3), arrowprops=dict(arrowstyle='->'))

    # Add labels on outliers
    for outlier in outliers:
        plt.annotate(f'{outlier:.2f}', xy=(1, outlier), xytext=(1.15, outlier))

plt.tight_layout()
plt.show()

# Caracterization of outliers on df1 (IQ method)

def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data < lower_bound) | (data > upper_bound)]

# Application on df1

numeric_cols = df1.select_dtypes(include=[np.number])
outliers = {}
for col in numeric_cols.columns:
    outliers[col] = detect_outliers_iqr(df1[col].dropna())
for col, values in outliers.items():
    print(f"Outliers for column {col}:")
    print(values)
    print("-" * 50)

"""# Exploratory Analysis

## Exploratory Analysis - Clinical_Dataset
"""

# Global histogramm for numerical values on df1

plt.figure(figsize=(40, 35))
for i, col in enumerate(numeric_cols):
    plt.subplot(6, 3, i+1)
    plt.hist(df1[col].dropna(), bins=20, alpha=0.7)
    plt.title(f'Histogram - {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Pie chart for number of sample by center on df2

center_counts = df2['CENTER'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(center_counts, labels=center_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
plt.title('Repartition of values on column Center')
plt.axis('equal')
plt.show()

"""## Exploratory analysis - BiologicalSample_Dataset"""

# Pie chart of sampled patient / unsampled patient as a function of Timepoint.

t0_columns = df2.iloc[:, 3:7]
t1_columns = df2.iloc[:, 7:11]
t2_columns = df2.iloc[:, 11:15]

# Function for counting

def count_oui_non(df):
    oui_count = (df == 'Yes').sum().sum()
    non_count = (df == 'No').sum().sum()
    return [oui_count, non_count]

# Application on df2

t0_counts = count_oui_non(t0_columns)
t1_counts = count_oui_non(t1_columns)
t2_counts = count_oui_non(t2_columns)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 3 graphiques côte à côte

labels = ['Yes', 'No']  # Légendes
colors = ['skyblue', 'lightcoral']  # Couleurs des sections

axes[0].pie(t0_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
axes[0].set_title('Ratio YES/NO at t0')

axes[1].pie(t1_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('Ratio YES/NO at t1')

axes[2].pie(t2_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
axes[2].set_title('Ratio YES/NO at t2')

plt.tight_layout()
plt.show()

# Pie chart of sampled patient / unsampled patient as a function of Timepoint and by sample type.

t0_columns = df2.iloc[:, 3:7]
t1_columns = df2.iloc[:, 7:11]
t2_columns = df2.iloc[:, 11:15]

# Function for counting.

def count_oui_non(df):
    counts = []
    for col in df.columns:
        oui_count = (df[col] == 'Yes').sum()
        non_count = (df[col] == 'No').sum()
        counts.append([oui_count, non_count])
    return counts

# Application on df2.

t0_counts = count_oui_non(t0_columns)
t1_counts = count_oui_non(t1_columns)
t2_counts = count_oui_non(t2_columns)
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
labels = ['Yes', 'No']
colors = ['skyblue', 'lightcoral']


for i in range(4):

    axes[0, i].pie(t0_counts[i], labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, i].set_title(f't0 - Sample {i+1}')

    axes[1, i].pie(t1_counts[i], labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, i].set_title(f't1 - Sample {i+1}')

    axes[2, i].pie(t2_counts[i], labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[2, i].set_title(f't2 - Sample {i+1}')

plt.tight_layout()
plt.show()

# Stacked bar chart for sample type by center.

sample_types = {
    'Swabs': df2.iloc[:, [3, 7, 11]].notnull().sum(axis=1),
    'Skin_biopsy': df2.iloc[:, [4, 8, 12]].notnull().sum(axis=1),
    'Whole_blood': df2.iloc[:, [5, 9, 13]].notnull().sum(axis=1),
    'Serum': df2.iloc[:, [6, 10, 14]].notnull().sum(axis=1)
}

samples_by_center = df2['CENTER'].to_frame()
for sample_type, counts in sample_types.items():
    samples_by_center[sample_type] = counts

samples_grouped = samples_by_center.groupby('CENTER').sum()

samples_grouped.plot(kind='bar', stacked=True, figsize=(12, 6), color=['skyblue', 'orange', 'green', 'red'])
plt.title('Repartition of sample type by center')
plt.xlabel('Center')
plt.ylabel('Number of sample')
plt.xticks(rotation=45)
plt.legend(title='Types of sample')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

"""# Question 1 - Longitudinal format"""

# Quality control/transformation for birth feature : replacing NaN values by 0 or 1 and transform to INT format.

df1['t0_Birth_year'] = df1['t0_Birth_year'].fillna(0).astype(int)
df1['t0_Birth_month'] = df1['t0_Birth_month'].fillna(1).astype(int)

# Concatenate birth_year and birth_month.

df1['Birth_date'] = pd.to_datetime(df1['t0_Birth_year'].astype(str) + '-' +
                                   df1['t0_Birth_month'].astype(str) + '-01', errors='coerce')

# Proposal of a first version for Longitudinal format.


# Transform the following columns.

columns_to_melt = [
                   'Name_Treatment_1_', 'Name_Treatment_2', 'Name_Treatment_3', 'Name_Treatment_4', 'Name_Treatment_5','Name_Treatment_6','Name_Treatment_7','Name_Treatment_8','Name_Treatment_9',
                   'ATC_code_Treatment_1', 'ATC_code_Treatment_2', 'ATC_code_Treatment_3', 'ATC_code_Treatment_4', 'ATC_code_Treatment_5', 'ATC_code_Treatment_6', 'ATC_code_Treatment_7', 'ATC_code_Treatment_8', 'ATC_code_Treatment_9',
                   'Allergic_rhinitis', 'Food_allergy', 'Asthma', 'Psoriasis', 'Diabetes_mellitus', 'Atopic_Dermititis',
                   'EASI_score', 'SCORAD_score',]

df_long = pd.DataFrame()

for col in columns_to_melt:
    melted = pd.melt(df1,
                     id_vars=['ID', 'Birth_date'],
                     value_vars=[f't0_{col}', f't1_{col}', f't2_{col}'],
                     var_name='Timepoint',
                     value_name=col)

    # Extract timepoint (t0, t1, t2) from columns.
    melted['Timepoint'] = melted['Timepoint'].str[:2]

    if df_long.empty:
        df_long = melted
    else:
        df_long = pd.merge(df_long, melted, on=['ID', 'Timepoint', 'Birth_date'])

# # Sort by ID and Timepoint.

df_long = df_long.sort_values(by=['ID', 'Timepoint']).reset_index(drop=True)
df_long.head()

# Check the general structure of the dataframe.

print(df_long.info())

# Check about behaviour of NaT values - Example of ID 715381.

df_filteredID = df_long[df_long['ID'] == 715381]
df_filteredID.head()

# Calculate age of the patient from 31/12/2024 and treatment of age outliers

reference_date = pd.to_datetime('2024-12-31')
df1['Age'] = ((reference_date - df1['Birth_date']).dt.days // 365)
df1['Age'] = df1['Age'].fillna(-1).astype(int)

# Proposal of a final version for Longitudinal format.


# Transform the following columns.

columns_to_melt = [
                   'Name_Treatment_1_', 'Name_Treatment_2', 'Name_Treatment_3', 'Name_Treatment_4', 'Name_Treatment_5','Name_Treatment_6','Name_Treatment_7','Name_Treatment_8','Name_Treatment_9',
                   'ATC_code_Treatment_1', 'ATC_code_Treatment_2', 'ATC_code_Treatment_3', 'ATC_code_Treatment_4', 'ATC_code_Treatment_5', 'ATC_code_Treatment_6', 'ATC_code_Treatment_7', 'ATC_code_Treatment_8', 'ATC_code_Treatment_9',
                   'Allergic_rhinitis', 'Food_allergy', 'Asthma', 'Psoriasis', 'Diabetes_mellitus', 'Atopic_Dermititis',
                   'EASI_score', 'SCORAD_score']

df_long = pd.DataFrame()

for col in columns_to_melt:
    melted = pd.melt(df1,
                     id_vars=['ID', 'Age'],
                     value_vars=[f't0_{col}', f't1_{col}', f't2_{col}'],
                     var_name='Timepoint',
                     value_name=col)

    # Extract timepoint (t0, t1, t2) from columns.
    melted['Timepoint'] = melted['Timepoint'].str[:2]

    if df_long.empty:
        df_long = melted
    else:
        df_long = pd.merge(df_long, melted, on=['ID', 'Timepoint','Age'])

# Sort by ID and Timepoint.

df_long = df_long.sort_values(by=['ID', 'Timepoint']).reset_index(drop=True)
df_long.head()

# Check for Age outliers

df_filteredID = df_long[df_long['ID'] == 715381]
df_filteredID.head()

"""# Question 2 - General characteristics of subset datas"""

# Filtering longitudinal dataframe on 'Atopic_Dermititis' column with both 'nein' or 'ja' values, excluding NaN values.

df_long_subset = df_long[df_long['Atopic_Dermititis'].isin(['nein', 'ja'])]
df_long_subset.head()

# Check the general structure of dataframe

print(df_long_subset.info())

# Creation of a "Patient characteristics" for numerical values.

timepoints = ['t0', 't1', 't2']
tables = {}

for timepoint in timepoints:
    subset = df_long_subset[df_long_subset['Timepoint'] == timepoint]

    # Clustering of subset.

    atopic = subset[subset['Atopic_Dermititis'] == 'ja']
    non_atopic = subset[subset['Atopic_Dermititis'] == 'nein']

    # Creation of the table and basic statistics calculation.
    summary = pd.DataFrame()

    summary['Age_Mean'] = [atopic['Age'].mean(), non_atopic['Age'].mean()]
    summary['Age_Std'] = [atopic['Age'].std(), non_atopic['Age'].std()]

    summary['EASI_Mean'] = [atopic['EASI_score'].mean(), non_atopic['EASI_score'].mean()]
    summary['EASI_Std'] = [atopic['EASI_score'].std(), non_atopic['EASI_score'].std()]

    summary['SCORAD_Mean'] = [atopic['SCORAD_score'].mean(), non_atopic['SCORAD_score'].mean()]
    summary['SCORAD_Std'] = [atopic['SCORAD_score'].std(), non_atopic['SCORAD_score'].std()]

    summary['Count'] = [atopic.shape[0], non_atopic.shape[0]]

    # Organization of the table and display

    summary['Group'] = ['ATOPIC DERMATITIS', 'NON ATOPIC DERMATITIS']
    summary = summary[['Group', 'Count', 'Age_Mean', 'Age_Std', 'EASI_Mean', 'EASI_Std', 'SCORAD_Mean', 'SCORAD_Std']]
    tables[timepoint] = summary

    print(f'{timepoint}:')
    print(summary)

# Creation of a "Patient characteristics" for categorical values.

timepoints = ['t0', 't1', 't2']
categorical_columns = ['Allergic_rhinitis', 'Food_allergy', 'Asthma', 'Diabetes_mellitus']

for timepoint in timepoints:
    subset = df_long_subset[df_long_subset['Timepoint'] == timepoint]

    # Clustering of subset.
    atopic = subset[subset['Atopic_Dermititis'] == 'ja']
    non_atopic = subset[subset['Atopic_Dermititis'] == 'nein']

    # Creation of the table and ratio calculation.
    categorical_summary = pd.DataFrame()
    for col in categorical_columns:
        atopic_counts = atopic[col].value_counts(normalize=True)
        non_atopic_counts = non_atopic[col].value_counts(normalize=True)

        categorical_summary.loc[col, 'AD_Yes'] = atopic_counts.get('ja', 0)
        categorical_summary.loc[col, 'AD_No'] = atopic_counts.get('nein', 0)
        categorical_summary.loc[col, 'Non_AD_Yes'] = non_atopic_counts.get('ja', 0)
        categorical_summary.loc[col, 'Non_AD_No'] = non_atopic_counts.get('nein', 0)

    # Display table.

    print(f'{timepoint}:')
    print(categorical_summary)

# Filtering of subset and check

df_nein = df_long[df_long['Atopic_Dermititis'] == 'nein']
df_ja = df_long[df_long['Atopic_Dermititis'] == 'ja']

df_nein.head()

"""## Data Visualisation

### Filter = 'Nein'
"""

# Histogram for numerical features.

columns = ['EASI_score', 'SCORAD_score', 'Age']

for col in columns:
    plt.figure(figsize=(8, 6))
    plt.hist(df_nein[col], bins=20, alpha=0.7, label='nein')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {col} with filter = nein')
    plt.legend()
    plt.grid(True)
    plt.show()

# Pie chart for Name_Treatment

columns = ['Name_Treatment_1_', 'Name_Treatment_2', 'Name_Treatment_3',
           'Name_Treatment_4', 'Name_Treatment_5', 'Name_Treatment_6',
           'Name_Treatment_7', 'Name_Treatment_8', 'Name_Treatment_9']

for col in columns:
    if col in df_nein.columns:
        data = df_nein[col].value_counts(dropna=False).nlargest(10)
        if not data.empty:
            plt.figure(figsize=(8, 6))
            plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, colors=plt.get_cmap('Set3').colors)
            plt.title(f'Distribution for the first 10th {col} with filter = nein')
            plt.tight_layout()
            plt.show()

# Pie chart for the ratio yes/no for comodities.

columns = ['Food_allergy', 'Asthma', 'Psoriasis', 'Diabetes_malletis']
for col in columns:
    if col in df_nein.columns:
        data = df_nein[col].value_counts(dropna=False).nlargest(10)
        if not data.empty:
            plt.figure(figsize=(8, 6))
            plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, colors=plt.get_cmap('Set3').colors)
            plt.title(f'Ratio for {col} with filter = nein')
            plt.tight_layout()
            plt.show()

# Histogramm for ATC Code

atc_columns = [f'ATC_code_Treatment_{i}' for i in range(1, 9)]
for col in atc_columns:
    if col in df_nein.columns:
        data = df_nein[col].value_counts(dropna=False).nlargest(10)
        if not data.empty:
            plt.figure(figsize=(10, 6))
            data.plot(kind='bar')
            plt.xlabel('Categories')
            plt.ylabel('Frequency')
            plt.title(f'Histogram for the repartition of the 10th first {col}')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

"""### Filter = 'Ja'"""

# Histogram for numerical features.

columns = ['EASI_score', 'SCORAD_score', 'Age']

for col in columns:
    plt.figure(figsize=(8, 6))
    plt.hist(df_ja[col], bins=20, alpha=0.7, label='ja', color='orange')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {col} with filter = ja')
    plt.legend()
    plt.grid(True)
    plt.show()

# Pie chart for Name_Treatment

columns = ['Name_Treatment_1_', 'Name_Treatment_2', 'Name_Treatment_3',
           'Name_Treatment_4', 'Name_Treatment_5', 'Name_Treatment_6',
           'Name_Treatment_7', 'Name_Treatment_8', 'Name_Treatment_9']

for col in columns:
    if col in df_ja.columns:
        data = df_ja[col].value_counts(dropna=False).nlargest(10)
        if not data.empty:
            plt.figure(figsize=(8, 6))
            plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, colors=plt.get_cmap('Set3').colors)
            plt.title(f'Distribution for the first 10th {col} with filter = ja')
            plt.tight_layout()
            plt.show()

# Pie chart for the ratio yes/no for comodities.
columns = ['Food_allergy', 'Asthma', 'Psoriasis', 'Diabetes_mellitus']
for col in columns:
    if col in df_ja.columns:
        data = df_ja[col].value_counts(dropna=False).nlargest(10)
        if not data.empty:
            plt.figure(figsize=(8, 6))
            plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, colors=plt.get_cmap('Set3').colors)
            plt.title(f'Ratio for{col} with filter = ja')
            plt.tight_layout()
            plt.show()

atc_columns = [f'ATC_code_Treatment_{i}' for i in range(1, 9)]
for col in atc_columns:
    if col in df_ja.columns:
        data = df_ja[col].value_counts(dropna=False).nlargest(10)
        if not data.empty:
            plt.figure(figsize=(10, 6))
            data.plot(kind='bar')
            plt.xlabel('Categories')
            plt.ylabel('Frequency')
            plt.title(f'Histogram for the repartition of the 10th first {col}')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

"""# Question 3 - Merging and specific filter/count"""

# In case when sampling is done in another center than the clinical follow-up, the couple (ID,CENTER)/Clinical follow-up could be different for the couple (ID,CENTER)/Sampling.
# QC need to be done between both dataset to avoid eventual issue during joining dataset.

comparison = (df1[['ID', 'CENTER']].sort_values(['ID', 'CENTER']).reset_index(drop=True) ==
              df2[['ID', 'CENTER']].sort_values(['ID', 'CENTER']).reset_index(drop=True))

all_equal = comparison.all().all()
print(" QC regarding (ID, CENTER) pairs OK? :", all_equal)

# Prepare df1 and df2 droping the index column.

df1 = df1.drop(columns=['Unnamed: 0'], errors='ignore')
df2 = df2.drop(columns=['Unnamed: 0'], errors='ignore')

# Merging dataset with the join on "ID Key"

df_merged = pd.merge(df1, df2, on='ID')
df_merged.head()

# Check global structure.

print(df_merged.info())

# Transform the following columns.
columns_to_melt = [
    'Name_Treatment_1_', 'Name_Treatment_2', 'Name_Treatment_3', 'Name_Treatment_4', 'Name_Treatment_5',
    'Name_Treatment_6', 'Name_Treatment_7', 'Name_Treatment_8', 'Name_Treatment_9',
    'ATC_code_Treatment_1', 'ATC_code_Treatment_2', 'ATC_code_Treatment_3', 'ATC_code_Treatment_4',
    'ATC_code_Treatment_5', 'ATC_code_Treatment_6', 'ATC_code_Treatment_7', 'ATC_code_Treatment_8', 'ATC_code_Treatment_9',
    'EASI_score', 'SCORAD_score', 'Skin_biopsy', 'Whole_blood'
]

df_final = pd.DataFrame()

for col in columns_to_melt:
    melted = pd.melt(df_merged,
                     id_vars=['ID'],
                     value_vars=[f't0_{col}', f't1_{col}', f't2_{col}'],
                     var_name='Timepoint',
                     value_name=col)

    # Extract timepoint (t0, t1, t2) from columns.
    melted['Timepoint'] = melted['Timepoint'].str[:2]

    if df_final.empty:
        df_final = melted
    else:
        df_final = pd.merge(df_final, melted, on=['ID', 'Timepoint'])

# Sort by ID and Timepoint.

df_final = df_final.sort_values(by=['ID', 'Timepoint']).reset_index(drop=True)
df_final.head(10)

# Check about dataframe structure.

print(df_final.info())

"""### Subquestion 3.1"""

# Inclusive filtering with EASI_score > 25, Timepoint = 't0', Skin_biopsy = 'Yes'.
filtered_df_t0 = df_final[
    (df_final['EASI_score'] > 25) &
    (df_final['Timepoint'] == 't0') &
    (df_final['Skin_biopsy'] == 'Yes')
]

# Exclusive filtering regarding Treatment_Name.
name_treatment_cols = ['Name_Treatment_1_'] + [f'Name_Treatment_{i}' for i in range(2, 9)]
filtered_df_t0 = filtered_df_t0[
    ~filtered_df_t0[name_treatment_cols].apply(lambda x: x.isin(['dupilumab', 'dupixent','dupilumap','dupilvmab'])).any(axis=1)
]

# Exclusive filtering regarding ATC_code_Treatment.
atc_code_cols = [f'ATC_code_Treatment_{i}' for i in range(2, 9)]
filtered_df_t0 = filtered_df_t0[
    ~filtered_df_t0[atc_code_cols].apply(lambda x: x.isin(['d11ah05'])).any(axis=1)
]

# Display results.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
filtered_df_t0.head()

# Inclusive filtering with EASI_score > 25, Timepoint = 't1', Skin_biopsy = 'Yes'.
filtered_df_t1 = df_final[
    (df_final['EASI_score'] > 25) &
    (df_final['Timepoint'] == 't1') &
    (df_final['Skin_biopsy'] == 'Yes')
]

# Exclusive filtering regarding Treatment_Name.
name_treatment_cols = ['Name_Treatment_1_'] + [f'Name_Treatment_{i}' for i in range(2, 9)]
filtered_df_t1 = filtered_df_t1[
    ~filtered_df_t1[name_treatment_cols].apply(lambda x: x.isin(['dupilumab', 'dupixent','dupilumap','dupilvmab'])).any(axis=1)
]

# Exclusive filtering regarding ATC_code_Treatment.
atc_code_cols = [f'ATC_code_Treatment_{i}' for i in range(2, 9)]
filtered_df_t1 = filtered_df_t1[
    ~filtered_df_t1[atc_code_cols].apply(lambda x: x.isin(['d11ah05'])).any(axis=1)
]

# Display results.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
filtered_df_t1.head()

# Inclusive filtering with EASI_score > 25, Timepoint = 't2', Skin_biopsy = 'Yes'.
filtered_df_t2 = df_final[
    (df_final['EASI_score'] > 25) &
    (df_final['Timepoint'] == 't2') &
    (df_final['Skin_biopsy'] == 'Yes')
]

# Exclusive filtering regarding Treatment_Name.
name_treatment_cols = ['Name_Treatment_1_'] + [f'Name_Treatment_{i}' for i in range(2, 9)]
filtered_df_t2 = filtered_df_t2[
    ~filtered_df_t2[name_treatment_cols].apply(lambda x: x.isin(['dupilumab', 'dupixent','dupilumap','dupilvmab'])).any(axis=1)
]

# Exclusive filtering regarding ATC_code_Treatment.
atc_code_cols = [f'ATC_code_Treatment_{i}' for i in range(2, 9)]
filtered_df_t2 = filtered_df_t2[
    ~filtered_df_t2[atc_code_cols].apply(lambda x: x.isin(['d11ah05'])).any(axis=1)
]

# Display results.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
filtered_df_t2.head()

"""### Subquestion 3.2"""

# Inclusive filtering with SCORAD_score < 50, Timepoint = 't0', Whole_blood = 'Yes'.
filteredbis_df_t0 = df_final[
    (df_final['SCORAD_score'] < 50) &
    (df_final['Timepoint'] == 't0') &
    (df_final['Whole_blood'] == 'Yes')
]

# Inclusive filtering regarding Treatment_Name.
name_treatment_cols = ['Name_Treatment_1_'] + [f'Name_Treatment_{i}' for i in range(2, 9)]
filteredbis_df_t0 = filteredbis_df_t0[
    filteredbis_df_t0[name_treatment_cols].apply(lambda x: x.isin(['dupilumab', 'dupixent', 'dupilumap', 'dupilvmab'])).any(axis=1)
]

# Inclusive filtering regarding ATC_code_Treatment.
atc_code_cols = [f'ATC_code_Treatment_{i}' for i in range(2, 9)]
filteredbis_df_t0 = filteredbis_df_t0[
    filteredbis_df_t0[atc_code_cols].apply(lambda x: x.isin(['d11ah05'])).any(axis=1)
]

# Display results.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
filteredbis_df_t0.head()

# Inclusive filtering with SCORAD_score < 50, Timepoint = 't1', Whole_blood = 'Yes'.
filteredbis_df_t1 = df_final[
    (df_final['SCORAD_score'] < 50) &
    (df_final['Timepoint'] == 't1') &
    (df_final['Whole_blood'] == 'Yes')
]

# Inclusive filtering regarding Treatment_Name.
name_treatment_cols = ['Name_Treatment_1_'] + [f'Name_Treatment_{i}' for i in range(2, 9)]
filteredbis_df_t1 = filteredbis_df_t1[
    filteredbis_df_t1[name_treatment_cols].apply(lambda x: x.isin(['dupilumab', 'dupixent', 'dupilumap', 'dupilvmab'])).any(axis=1)
]

# Inclusive filtering regarding ATC_code_Treatment.
atc_code_cols = [f'ATC_code_Treatment_{i}' for i in range(2, 9)]
filteredbis_df_t1 = filteredbis_df_t1[
    filteredbis_df_t1[atc_code_cols].apply(lambda x: x.isin(['d11ah05'])).any(axis=1)
]

# Display results.
pd.set_option('display.max_rows', None)  # Afficher toutes les lignes
pd.set_option('display.max_columns', None)  # Afficher toutes les colonnes
filteredbis_df_t1.head()

# Inclusive filtering with SCORAD_score < 50, Timepoint = 't2', Whole_blood = 'Yes'.
filteredbis_df_t2 = df_final[
    (df_final['SCORAD_score'] < 50) &
    (df_final['Timepoint'] == 't2') &
    (df_final['Whole_blood'] == 'Yes')
]

# Inclusive filtering regarding Treatment_Name.
name_treatment_cols = ['Name_Treatment_1_'] + [f'Name_Treatment_{i}' for i in range(2, 9)]
filteredbis_df_t2 = filteredbis_df_t2[
    filteredbis_df_t2[name_treatment_cols].apply(lambda x: x.isin(['dupilumab', 'dupixent', 'dupilumap', 'dupilvmab'])).any(axis=1)
]

# Inclusive filtering regarding ATC_code_Treatment.
atc_code_cols = [f'ATC_code_Treatment_{i}' for i in range(2, 9)]
filteredbis_df_t2 = filteredbis_df_t2[
    filteredbis_df_t2[atc_code_cols].apply(lambda x: x.isin(['d11ah05'])).any(axis=1)
]

# Display results.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
filteredbis_df_t2.head()
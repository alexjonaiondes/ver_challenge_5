# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 16:51:06 2026

@author: Alex John
"""

import os
import chardet
import pandas as pd
import numpy as np
import Levenshtein as levenshtein
from difflib import SequenceMatcher as SM



'''
# Open the file in read mode ('r')
with open('presales_data_sample.csv', 'r', encoding='utf-8') as file:
    # Read the entire content into a string variable
    csv_as_text = file.read()
    count = file.read().count(';')
'''

'''
# Open in binary mode ('rb')
with open('presales_data_sample.csv', 'rb') as file:
    # Read the first 1,000,000 bytes for a quick check
    raw_data = file.read(1000000)
    result = chardet.detect(raw_data)
'''

initial_df = pd.read_csv('presales_data.csv', sep=';', encoding='utf-8')
df = pd.read_csv('presales_data.csv', sep=';', encoding='utf-8')

#create a list of all columns
col_list = df.columns.tolist()

# create input_cols list
input_cols = [col for col in df.columns if 'input' in col]

# Convert col_list to a set for O(1) lookup speed
col_match_set = set(col_list)

# Create a dictionary mapping: {'input_NAME': 'NAME'}
col_matches = {col: col[6:] for col in input_cols if col[6:] in col_match_set}

# Create new input_column 
df['input_column'] = df[input_cols].fillna('').astype(str).agg(''.join, axis=1)

# Build input and output string for comparison
# Make list for input and output cols matched
input_col_list = list(col_matches.keys())
output_col_list = list(col_matches.values())

# Check if input is valid AND output is valid. 
# We use .values to ignore column names and compare by position.
valid_mask = df[input_col_list].notna().values & df[output_col_list].notna().values

# Apply mask: Keep valid values, replace invalid pairs with empty string ''
clean_inputs = np.where(valid_mask, df[input_col_list], '')
clean_outputs = np.where(valid_mask, df[output_col_list], '')

# Concatenate the cleaned arrays row-by-row
# We wrap in DataFrame just to use the convenient .agg method
df['input_string'] = pd.DataFrame(clean_inputs).astype(str).agg(''.join, axis=1)
df['output_string'] = pd.DataFrame(clean_outputs).astype(str).agg(''.join, axis=1)

# Regex explanation:
# \W matches anything that is NOT a letter, number, or underscore (including whitespace & punctuation)
# _ matches the underscore explicitly (often treated as a word char, so we add it to remove it)
df['clean_input_string'] = df['input_string'].str.lower().str.replace(r'[\W_]+', '', regex=True)
df['clean_output_string'] = df['output_string'].str.lower().str.replace(r'[\W_]+', '', regex=True)

#create new dataframe from processed columns just to visually check them
df_processed = df[['input_string','clean_input_string','output_string','clean_output_string']]

#calculating levenshtein distance and ratio for processed strings
df['lev_distance'] = [levenshtein.distance(a,b) for a, b in zip(df['clean_input_string'], df['clean_output_string'])]
df['lev_ratio'] = [levenshtein.ratio(a,b) for a, b in zip(df['clean_input_string'], df['clean_output_string'])]

#get companies with exact name. It seems there are 42 of them. Great, I found the meaning of life.
df_exact = df[df['lev_ratio'] == 1]

#need to drop companies that have exact names
values_to_drop = df.loc[df['lev_ratio'] == 1, 'input_row_key']
df_nonclean = df[~df['input_row_key'].isin(values_to_drop)]

#testing if it dropped the correct ones
df_exact_test = df_nonclean[df_nonclean['lev_ratio'] == 1]


#counting number of rows with levenshtein distance larger than a few values
count_lev_9 = (df_nonclean['lev_ratio'] > 0.9).sum()
count_lev_8 = (df_nonclean['lev_ratio'] > 0.8).sum()
count_lev_7 = (df_nonclean['lev_ratio'] > 0.7).sum()


#calculating similarity ratio based on Ratcliff/Obershelp string matching algorithm
df_nonclean['similarity_ratio'] = [SM(None, a, b).ratio() for a, b in zip(df_nonclean['clean_input_string'], df_nonclean['clean_output_string'])]


#counting number of rows with similarity ratio larger than a few values
count_sim_9 = (df_nonclean['similarity_ratio'] > 0.9).sum()
count_sim_8 = (df_nonclean['similarity_ratio'] > 0.8).sum()
count_sim_7 = (df_nonclean['similarity_ratio'] > 0.7).sum()


#filtering the dataset by similarity ratio >= 0.7
sub_dfnonclean = df_nonclean[df_nonclean['similarity_ratio'] >= 0.7 ]
#sub_dfnonclean2 = df_nonclean[df_nonclean['similarity_ratio'] >= 0.8 ]
#sub_dfnonclean2 = df_nonclean[df_nonclean['lev_ratio'] >= 0.8 ]

#finding companies with similarity ratio >= 0.7
idx = sub_dfnonclean.groupby('input_row_key')['similarity_ratio'].idxmax()
sub_dfsim = sub_dfnonclean.loc[idx]

#finding companies with similarity ratio >= 0.8
#idx = sub_dfnonclean2.groupby('input_row_key')['similarity_ratio'].idxmax()
#sub_dfsim2 = sub_dfnonclean2.loc[idx]

#delete useless variables
del clean_inputs, clean_outputs, col_list, col_match_set, col_matches, idx, valid_mask, values_to_drop

#merge dataset with exact companies and dataset with >= 0.8 similar companies
#clean_dataset = pd.concat([df_exact, sub_dfsim], ignore_index = True)

#merge dataset with exact companies and dataset with >= 0.7 similar companies
clean_dataset = pd.concat([df_exact, sub_dfsim], ignore_index = True)
clean_dataset.loc[clean_dataset['lev_ratio']==1, 'similarity_ratio'] = 1

columns_to_drop = clean_dataset.iloc[:, np.r_[0:9, 76:83]].columns.tolist()

companies_data = clean_dataset.drop(columns=columns_to_drop)
companies_data = companies_data.sort_values('similarity_ratio', ascending=False)

#count_sim = (clean_dataset['similarity_ratio'] != clean_dataset['lev_ratio']).sum()

#clean_dataset.to_csv('companies_08.csv', index=False, encoding='utf-8-sig')

companies_data.to_csv('companies_data.csv', index=False, sep=';', encoding='utf-8-sig')





# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 14:43:27 2026

@author: Alex John
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px



initial_df = pd.read_csv('companies_data.csv', sep=';', encoding='utf-8-sig')
df = pd.read_csv('companies_data.csv', sep=';', encoding='utf-8-sig')

#check if companies are unique by veridion_id column
duplicates = df[df.duplicated('veridion_id', keep=False)]
count_duplicates = duplicates['veridion_id'].nunique()

#drop duplicates 
df = df.drop_duplicates(subset=['veridion_id'], keep='first')

#reindexing dataframe
df = df.reset_index(drop=True)

#checking values for employee number and revenue to see how many are NaN
valid_revenue = df['revenue'].notna().mean() * 100
valid_employee_count = df['employee_count'].notna().mean() * 100
'''
#generating a chart that shows the number of companies for each country
counts = df['main_country'].value_counts().reset_index()
counts.columns = ['main_country', 'company_name']
fig = px.bar(counts, x='main_country', y='company_name', color='company_name', title="Companies by Country", 
             labels={'main_country': 'Country', 'company_name': 'Number of companies'})

fig.write_html("companies_per_country.html")

'''



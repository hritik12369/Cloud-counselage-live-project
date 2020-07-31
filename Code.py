import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import StringIO
from collections import Counter
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher


def Recommender_System():
    employee_df = pd.read_csv('employee.csv')
    for col in employee_df.columns:
        employee_df[col] = employee_df[col].str.lower()
        if col=='Event1' or col=='Event2':
            employee_df[col] = [val[:-1] for val in employee_df[col]]
    
    columns_keyword = {}
    for column in employee_df.columns:
        columns_keyword[column] = [nlp(text) for text in employee_df[column].dropna(axis=0)]
        
    df_text = pd.read_csv('Input_Text.csv')
    
    recommendation = pd.DataFrame(columns=['Entered Text', 'Recommended Employee'])
        
    for original_text in df_text['Entered Text']:
        #original_text = input('Enter Text:- ')
        text = original_text.lower()
        
        matcher = PhraseMatcher(nlp.vocab)
        for column in employee_df.columns:
            matcher.add(column, None, *columns_keyword[column])

        doc = nlp(text)
        d = []  
        matches = matcher(doc)
        for match_id, start, end in matches:
            rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
            span = doc[start : end]  # get the matched slice of the doc
            d.append((rule_id, span.text))      
        keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())

        df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
        df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
        df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
        df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
        df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
        dataf = pd.concat([df3['Subject'], df3['Keyword']], axis = 1)
        dataf = dataf.drop_duplicates('Keyword')

        domains = list(i.strip() for i in dataf['Keyword'][dataf['Subject']=='Domain'])
        #print(domains)
        if len(domains)==0:
            domains = ['other']

        events = list(i.strip() for i in dataf['Keyword'][dataf['Subject']!='Domain'])
        
        #recommendation = pd.DataFrame(columns=['Entered Text', 'Recommended Employee'])
        
        if len(events) > 0:
            employees = []
            for domain in domains:
                matched_domain = employee_df[employee_df['Domain']==domain]
                #print(matched_domain)
                for event in events:
                    employee_name_1 = list(matched_domain['Name'][matched_domain['Event1']==event])
                    employee_name_2 = list(matched_domain['Name'][matched_domain['Event2']==event])
                    employees += employee_name_1 + employee_name_2

            matched_employee = [employee.title() for employee in employees]
            matched_employee = list(set(matched_employee))
            matched_employee = " ".join(str(elem+",") for elem in matched_employee)
            matched_employee = matched_employee[:-1]
            
            recommendation = recommendation.append({'Entered Text' : original_text, 'Recommended Employee': matched_employee}, ignore_index=True)
            
        else:
            recommendation = recommendation.append({'Entered Text' : original_text, 'Recommended Employee': 'None'}, ignore_index=True)
        
        recommendation.to_excel('output.xlsx', index=False)


Recommender_System()





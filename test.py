
import streamlit as st
import pandas as pd
from streamlit.components.v1 import html
import re

st.markdown("# Preprocessing")  
st.markdown("<br>",unsafe_allow_html=True)
data = st.file_uploader(label="Upload dataset")
if data is not None:
    try:
        data = pd.read_csv(data)
    except:
        st.error("Can only receive inputs of .csv type")
        quit()


def select_columns(data):
    with st.form("columns"):
        for column in data.columns:
            if re.match("customerID", column) or re.match("Churn", column):
                continue
            selectbox = st.checkbox(column)
            if selectbox:
                variables.append(column)
        submitted = st.form_submit_button("Submit")
    return submitted

def var_data_types(data):
    with st.form("data_type"):
        for column in variables:
            selectbox = st.selectbox(column, ('Select type','Numerical', 'Ordinal', 'Categorical'))
            if selectbox != 'Select type':
                variable_data_type.update({column: selectbox})
        submitted = st.form_submit_button("Submit")
    return submitted

def var_format(data):
    with st.form("data_format"):
        for column in variables:
            selectbox = st.text_input(column)
            if len(selectbox) > 0:
                variable_format.update({column: selectbox})
        submitted = st.form_submit_button("Submit")
    return submitted


variables = []
variable_data_type = {}
variable_format = {}

if data is not None and "Churn" in data.columns and "customerID" in data.columns:
    if len(data.columns) <= 2:
        st.warning("Data contains insufficient columns")
    else:
        st.markdown("Your input")
        st.dataframe(data)

        st.markdown("Select the columns you want to use")
        submitted = select_columns(data)
        if len(variables) > 0:
            st.write(variables)
            st.markdown("Specify the type of data each variable is")
            submitted = var_data_types(data)
            st.write(variable_data_type)

            if len(variable_data_type) == len(variables):
                st.markdown("Specify the regex format for each variable (*Optional*)")
                submitted = var_format(data)
                st.write(variable_format)
                
                if len(variable_format) == len(variables) or submitted:
                    st.markdown("<br>",unsafe_allow_html=True)
                    st.markdown("# EDA (Exploratory Data Analysis)")
            else:
                st.error("Fill in all fields")
        else:
            st.error("Select at least one variable")
            
elif data is not None and "Churn" not in data.columns:
    st.warning("Error: No column named 'Churn'")
elif data is not None and "customerID" not in data.columns:
    st.warning("Error: No column named 'customerID'")
else:
    st.warning("No input file")
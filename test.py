
import streamlit as st
import streamlit.components.v1 as componentsx
from pprint import pprint 
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import metrics
import regex as re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from matplotlib.colors import ListedColormap
from sklearn import linear_model
import sys 
from patsy import dmatrices
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import churn_functions as c
from sklearn import tree
from sklearn import svm
import statsmodels.formula.api as smf
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import RocCurveDisplay
import random 
from bokeh.plotting import figure
# CHANGE IF NOT LIKE
st.set_page_config(layout="wide")

st.markdown("""<h1>Churn Model</h1>""", unsafe_allow_html=True)  
st.markdown("<br>",unsafe_allow_html=True)
data = st.file_uploader(label="Upload dataset")
if data is not None:
    try:
        data = pd.DataFrame(pd.read_csv(data, encoding="unicode_escape"))
    except:
        st.error("Error: Can only receive inputs of .csv type",  icon="🛑")
        quit()
    with st.sidebar:
        st.markdown("# Churn Model")
        st.markdown(f"""<br>
        <a href=#preprocessing style="text-decoration: none;color: black;">Preprocessing</a>
        """,
        unsafe_allow_html=True)
        st.markdown(f"""<br>
        <a href=#eda-exploratory-data-analysis style="text-decoration: none;color: black;">EDA (Exploratory Data Analysis)</a>
        """,
        unsafe_allow_html=True)
        st.markdown(f"""<br>
        <a href=#splitting-the-data style="text-decoration: none;color: black;">Splitting the Data</a>
        """,
        unsafe_allow_html=True)
        st.markdown(f"""<br>
        <a href=#model-results style="text-decoration: none;color: black;">Model Results</a>
        """,
        unsafe_allow_html=True)
        st.markdown(f"""<br>
        <a href=#dashboard style="text-decoration: none;color: black;">Dashboard</a>
        """,
        unsafe_allow_html=True)
        st.markdown(f"""<br>
        <a href=#making-predictions style="text-decoration: none;color: black;">Making predictions</a>
        """,
        unsafe_allow_html=True)
       
       

def select_columns(data, customerID, churn):
    with st.form("columns"):
        for column in data.columns:
            if re.match(customerID, column) or re.match(churn, column):
                continue
            selectbox = st.checkbox(column)
            if selectbox:
                variables.append(column)
        submitted = st.form_submit_button("Submit")
    return submitted


# INCLUDE INDEX?????
def var_data_types(data):
    with st.form("data_type"):
        for column in variables:
            dtype = data[column].dtypes
            #st.write(dtype)
            if dtype != "object":
                default_ix = 1
            else:
                default_ix = 3
            selectbox = st.selectbox(column, ('Select type','Numerical', 'Ordinal', 'Categorical'), index=default_ix, key=f"1{column}")
            if selectbox != 'Select type':
                variable_data_type.update({column: selectbox})
        submitted = st.form_submit_button("Submit")
    return submitted

# FIX LATER

def var_format(data, variable_data_type):
    with st.form("data_format"):
        for column in variables:
            #options = st.multiselect(column, data[column].unique(), data[column].unique())
           # if variable_data_type[column] == "Numerical":
                
            selectbox = st.text_input(column)
            if len(selectbox) > 0:
                variable_format.update({column: selectbox})
        submitted = st.form_submit_button("Submit")
    return submitted


variables = []
variable_data_type = {}
variable_format = {}
all_models = []
model_names = ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Support Vector Machine']
model_metrics = []

#if data is not None:
 #   try:
   #     churn = [i for i in data.columns if bool(re.match("[cC]hurn", i))][0]
  #  except:
    #    pass
    #is_churn_col = bool([i for i in data.columns if bool(re.match("[cC]hurn", i))])
        

if data is not None:
    if len(data.columns) <= 2:
        st.error("Error: Data contains insufficient columns",  icon="🛑")
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Input", "Preprocessing", "Building the model", "Model results", "Investigation of model", "Making predictions"])
        with tab1:
            
            st.info("Your input", icon="ℹ️")
            st.dataframe(data)
            st.markdown("<br>",unsafe_allow_html=True)
        with tab2:
            selection = [i for i in data.columns]
            selection.insert(0, "Select column")
            churn = st.selectbox("Select the churn column", selection)
            if churn == "Select column":
                quit()
            customerID = st.selectbox("Select the customer primary key of the data", selection)
            if customerID == "Select column":
                st.error("Error: Invalid ID", icon="🛑")
                quit()
            data = data.drop_duplicates(subset = [customerID], keep='first')
            original_data = data
            all_customer_ID = data[customerID]
            if not all_customer_ID.is_unique:
                st.error("Error: Not a primary key", icon="🛑")
                quit()
            st.markdown("<br>",unsafe_allow_html=True)
            st.markdown("Select the columns you want to use")
            submitted = select_columns(data, customerID, churn)
            data.drop(variables, axis=1)
            if len(variables) == 0:
                st.error("Error: Select at least one variable",icon="🛑")
                quit()
            #st.write(variables)
            st.markdown("Specify the type of data each variable is")
            submitted = var_data_types(data)
            #st.write(variable_data_type)

            # LAST CHANGE
            if len(variable_data_type) != len(variables):
                st.error("Error: Select at least one variable",  icon="🛑")
                quit()
            
            # Check if there are any categorical / ordinal variables with too many unique values
            too_many_vars = False
            too_many = []
            for i in data.columns:
                if i not in variable_data_type:
                    continue
                if variable_data_type[i] == "Categorical":
                    poss = len(data[i].unique())
                    if poss > 100:
                        too_many_vars = True
                        too_many.append(i)
            if too_many_vars:
                st.warning(f"Warning: These following categorical variables: [{','.join(too_many)}] will produce too many columns in the new dataset. As a result, the analysis will take significantly longer to output. Make sure to check whether the variable type is correct or if you want to include this variable at all.", icon="⚠️")
                quit_opt = st.selectbox('Continue running?',('No', 'Yes'))
                if quit_opt == "No":
                    quit()


            #if len(variable_data_type) == len(variables):
                

                # READ THIS !!!!!
                #     df["Scale"] = df["Score"].replace(scale_mapper)

                # Ordinal variables 
            
            ordinal_rank = {}
            if len([i for i in variable_data_type.values() if i == "Ordinal"]) > 0:
                st.markdown("For all ordinal variables, select the hierachical order of values")
                ordinal_vars = [(key, data[key].unique()) for key, value in variable_data_type.items() if value == "Ordinal"]
                with st.form("ordinal"):
                    for vars in ordinal_vars:
                        #st.markdown(vars[0])
                        ranks = {}
                        for values in vars[1]:
                            number = st.selectbox(f"{values}", [i for i in range(len(vars[1]))])
                            ranks.update({values: int(number)})
                        ordinal_rank.update({vars[0]: ranks})
                    submitted = st.form_submit_button("Submit")
                default_order = []
                for var, vals in ordinal_rank.items():
                    check_order = []
                    for rank in vals:
                        check_order.append(vals[rank])
                    #st.write(check_order)
                    if len(set(check_order)) != len(check_order):
                        default_order.append(var)

                #st.write(default_order)
                if len(default_order) > 0:
                    st.warning("Warning: You have duplicate orders. Ordinal scale is reset to the default encoding", icon="⚠️")
                    for default_ord in default_order:
                        orders = ordinal_rank[default_ord]
                        i = 0
                        for pos in orders:
                            orders[pos] = i
                            i += 1
                        
            ordinal_rank.update({churn: {"No": 0, "Yes": 1}})

            #st.write(ordinal_rank)
            
            #st.markdown("Specify the regex format or a list of possible values seperated by commas for each variable (*Optional*)")
            variables.append(churn)
            #submitted = var_format(data, variable_data_type)
            #st.write(variable_format)
            #if len(variable_format) == len(variables):
            data = data[variables]

            st.markdown("<br>",unsafe_allow_html=True)
            #st.divider()
            st.write("---")
            st.markdown("# EDA (Exploratory Data Analysis)")
            st.markdown("<br>",unsafe_allow_html=True)
            st.markdown("### Distribution of variables ###")
            st.markdown("<br>",unsafe_allow_html=True)
            st.write(data.describe(include = 'all'))
            st.markdown("<br>",unsafe_allow_html=True)
                    

            st.markdown("### Barplot for selected variable ###")
            st.markdown("<br>",unsafe_allow_html=True)
            st.info("Barplots can help provide insight into a variable's frequency distribution and identifying any anomalies or outliers in the data", icon="ℹ️")
            variable = st.radio("Select a variable", variables)
            col1, col2 = st.columns(2)
            with col1:
                st.write(variable)
                st.write(data[variable].value_counts())
            with col2:
                #fig = plt.figure()
                #data[variable].value_counts().plot(kind='bar')
                #st.pyplot(fig) 
                st.markdown(variable)    
                st.bar_chart(pd.DataFrame(data[variable].value_counts())) 
            

            
            st.markdown("### Scatterplot for selected variable ###")
            st.markdown("<br>",unsafe_allow_html=True)
            st.info("Scatterplots can elucidate any degree and direction of correlation between two variables", icon="ℹ️")

            st.markdown("Select two variables for the plot")
            col1, col2 = st.columns(2, gap = "large")
            with col1:
                x_scplt = st.radio("X axis", variables)
            with col2:
                y_scplt = st.radio("Y axis", variables)
            col1, col2 = st.columns([3, 2])
            with col1:
                fig = plt.figure()
                plt.scatter(data[x_scplt], data[y_scplt])
                plt.xlabel(x_scplt)
                plt.ylabel(y_scplt)
                plt.title(f"{y_scplt} vs {x_scplt}")
                st.pyplot(fig)
            


            
            st.markdown("<br>",unsafe_allow_html=True)
            st.markdown("### Cleaning the data")
            st.markdown("<br>",unsafe_allow_html=True)
            vif_thres_test = st.slider("Select a VIF threshold for highly correlated variables", 0, 100, 10)
            col1, col2, col3 = st.columns(3, gap="Small")
            with col1:
                st.info("VIF (Variance Inflation Factor) is a measure of multicollinearity of independent variables that is the measure of correlations between variables.", icon="ℹ️")
            with col2:
                st.info("A high VIF corresponds to a significant correlation between variables and is unfavourable as it makes it difficult to distinguish their individual effects on the dependent variable", icon="ℹ️")
            with col3:
                st.info("The default VIF threshold is set at 10.", icon="ℹ️")
            propo_thres_test = st.slider("Select a threshold for dropping columns that exceed a certain percentage of nulls", 0.0, 1.0, 0.5)
            st.info("The default threshold for dropping columns by their constituent percentage of nulls is 0.5", icon="ℹ️")
            sd_threshold = st.slider("Select a standard deviation that would indicate an outlier", 0.0, 5.0, 3.0)   
            col1, col2 = st.columns(2, gap="Small")
            with col1:   
                st.info(f"Around {round(norm.cdf(sd_threshold)*100, 2)}% of the dataset will fall within {sd_threshold} standard deviations of the dataset", icon="ℹ️")
            with col2:
                st.info("The default standard deviations for outliers is set as 3.0", icon="ℹ️")
            dispro_thres_test = st.slider("Select a threshold for dropping columns that have values that make up most of the column values", 0.0, 1.0, 0.9)
            col1, col2 = st.columns(2, gap="Small")
            with col1:
                st.info("Heavily skewed columns may lead to biased estimates and incorrect conclusions", icon="ℹ️")
            with col2:
                st.info("The default threshold to drop columns when a value dominates 90% of the column", icon="ℹ️") 
            st.markdown("<br>",unsafe_allow_html=True)


            data_vif = c.clean_data(data, True, vif_thres_test, propo_thres_test, dispro_thres_test, all_customer_ID, variable_data_type, customerID, sd_threshold, ordinal_rank, churn)
            data = data_vif[1]
            to_remove, to_purge,to_drop, curr_customer_ID, removed_cust = data_vif[2], data_vif[3], data_vif[4], data_vif[5], data_vif[7]
            not_scaled = data_vif[6]
            not_encoded = data_vif[8]

            
            st.markdown("### Summary of cleaning output",unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)

            if len(to_remove) > 0:
                st.markdwon(f"-  Column/s that exceed the given proportion of nulls {propo_thres_test} is/are {' and '.join(to_remove)}")
            else:
                st.markdown(f"-  No column has excessive null entries by given threshold of {propo_thres_test}",unsafe_allow_html=True)

            if len(to_purge) > 0:
                st.markdown(f"-  Column/s that are heavily skewed with the maximum skew being {dispro_thres_test} is/are {' and '.join(to_purge)}")
            else:
                st.markdown(f"-  No column is heavily skewed with the maximum skew being {dispro_thres_test}",unsafe_allow_html=True)
            if len(removed_cust) > 0:
                st.markdown(f"-  Removed row/s that have many outliers is/are those with customerIDs: {' and '.join(removed_cust)}")
            else:
                st.markdown(f"-  No rows were removed by the count of outliers within each row",unsafe_allow_html=True)
            if len(to_drop) > 0:
                st.markdown(f"-  Variable/s which exceed the given threshold VIF of {vif_thres_test} is/are {' and '.join(to_drop)}",unsafe_allow_html=True)
            else:
                st.markdown(f"-  No variables exceed the given threshold VIF of {vif_thres_test}",unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)


            # TODO 
            removed = to_remove + to_purge + to_drop
            if len(removed) > 0:
                st.markdown("Keep any of the discarded variables?")
                for var in removed:
                    selectbox = st.checkbox(var)
                st.markdown("<br>",unsafe_allow_html=True)

            st.markdown("### Correlations between variables ###")
            st.markdown("<br>",unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap = "large")
            with col1:
                st.markdown("Correlation Heatmap")
                fig, ax = plt.subplots()
                matrix = data.corr()
                sns.heatmap(matrix, annot=True, ax = ax)
                st.write(fig)
                col_1, col_2 = st.columns(2, gap="small")
                with col_1:
                    st.info("The correlation heatmap shows the correlation coefficient of between each variable where higher values relates to greater degrees of correlation", icon="ℹ️")
                with col_2:
                    st.info("Generally, correlation coefficients greater than 0.7 are considered to be strong correlations", icon="ℹ️")
                st.info("We use this heatmap to remove independent variables that are correlated to each other", icon="ℹ️")
            with col2:
                st.markdown("VIF of each variable")
                #vif_df = c.vif_df(data)
                st.dataframe(data_vif[0])
                st.info("Entries in VIF are invisible if their values are close to infinity", icon="ℹ️")
            st.markdown("<br>",unsafe_allow_html=True)
            st.markdown("### Data after cleaning")
            st.markdown("<br>",unsafe_allow_html=True)
            st.dataframe(data)

        with tab3:
            st.markdown("# Splitting the Data #")
            st.markdown("<br>",unsafe_allow_html=True)
            st.info("We split the dataset into a training set and a test set, to evaluate how well our machine learning model performs on each split", icon="ℹ️")
            split_size = st.slider("Select a training size split", 0.0, 1.0, 0.75)
            if split_size == 1:
                st.error("Error: Training set cannot be the entire dataset.",icon="🛑")
                quit()
            elif split_size == 0:
                st.error("Error: Training set cannot be none.",icon="🛑")
                quit()
            num_cols = len(data.columns)
            x = data.iloc[:, :-1]
            y = data.iloc[:, num_cols - 1]

            st_x = StandardScaler()
            x_scaled = st_x.fit_transform(x)
            x = pd.DataFrame(x_scaled, index=x.index, columns=x.columns)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-split_size, random_state=1)
            topcol1, topcol2 = st.columns(2, gap = "large")
            with topcol1:
                st.markdown("### Training set ###")
                col1, col2 = st.columns(2, gap = "large")
                with col1:
                    st.markdown("X-train")
                    st.dataframe(x_train)
                with col2:
                    st.markdown("Y-train")
                    st.dataframe(y_train)
            with topcol2:
                st.markdown("### Test set ###")
                col1, col2 = st.columns(2, gap = "large")
                with col1:
                    st.markdown("X-test")
                    st.dataframe(x_test)
                with col2:
                    st.markdown("Y-test")
                    st.dataframe(y_test)

            st.markdown("<br>",unsafe_allow_html=True)
            

            bestfeatures = c.feat_sel1(data, x, x_train, x_test, y_train, y_test)
            model_by_Anova = bestfeatures[1]
            bestfeatures = bestfeatures[0]
            #fig = plt.figure()
            #pyplot.bar([i for i in range(len(bestfeatures))], bestfeatures)
            st.markdown("### Feature Selection ###")
            st.markdown("<br>",unsafe_allow_html=True)
            #plt.title("ANOVA f-test Feature Selection")
            #st.pyplot(fig)
            #lol["Names"] = [i for i in model_by_Anova]
            feature_plot = pd.DataFrame(index = [i for i in data.columns[:-1]])    
            feature_plot["Variables"] = bestfeatures
            st.markdown("ANOVA f-test Feature Selection")   
            st.markdown("<br>",unsafe_allow_html=True)
            st.bar_chart(feature_plot)
            
            #st.write(data.columns)
            predictors = set([i for i in data.columns if i != churn])

            selected_pred = [i for i in model_by_Anova]
            selected_indices = [i for i in model_by_Anova.values()]

           
            if len(selected_pred) == len(predictors):
                col1, col2 = st.columns(2, gap="small")
                with col1:
                    st.info(f"Using the F-statistic, we can calculate the p-value and the variables that yields a p-value of less than 0.05 are {', '.join(selected_pred)}", icon="ℹ️")
                with col2:
                    st.info("No variables were discarded", icon="ℹ️")
            else:
                discarded = predictors.difference(selected_pred)
                st.info(f"Using the F-statistic, we can calculate the p-value and the variables that yields a p-value of less than 0.05 are {', '.join(selected_pred)} excluding {' '.join(discarded)}", icon="ℹ️")
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("Keep any of the discarded variables?")
                for var in discarded:
                    selectbox = st.checkbox(var, key=f"2{var}")
                    index = [i for i in data.columns].index(var)
                    selected_indices.append(index)
                    model_by_Anova.update({var: index})
                st.markdown("<br>",unsafe_allow_html=True)




            
            x_train = x_train.iloc[:, selected_indices]
            x_test = x_test.iloc[:, selected_indices]  
            st.markdown("### After Feature Selection ###")
            st.info("This is the final training set and test set we will be using for building our model", icon="ℹ️")

            st.markdown("<br>",unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap = "large")
            with col1:
                st.markdown("Modified X-train")
                st.dataframe(x_train)
            with col2:
                st.markdown("Modified X-test")
                st.dataframe(x_test)
        with tab4:
            st.markdown("# Model Results #")
            st.markdown("<br>",unsafe_allow_html=True)
            st.info("We select the best model from the built machine learning models below.", icon="ℹ️")
            model_type = st.radio("Select to view each model", ('Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Support Vector Machine'))

    #[LogisticRegression(random_state=0,)
            all_classifiers = [LogisticRegression(random_state=0), GaussianNB(), tree.DecisionTreeClassifier(), svm.SVC(probability=True)]
            for classifier in all_classifiers:
                classifier.fit(x_train, y_train)
                cm_model = c.get_cm(data, x, x_train, x_test, y_train, y_test, classifier)
                all_models.append(cm_model)
                all_metrics = c.get_metrics(y_test, cm_model[2]) 
                model_metrics.append(all_metrics)

            best_scores = [[i for i in model["Scores"]] for model in model_metrics]
            ranked_acc = {k[1]: l for l, k in enumerate(sorted([(i[0], j) for j, i in enumerate(best_scores)], key = lambda x: x[0], reverse=True))}
            ranked_pre = {k[1]: l for l, k in enumerate(sorted([(i[1], j) for j, i in enumerate(best_scores)], key = lambda x: x[0], reverse=True))}
            ranked_rec = {k[1]: l for l, k in enumerate(sorted([(i[2], j) for j, i in enumerate(best_scores)], key = lambda x: x[0], reverse=True))}
            ranked_f1s = {k[1]: l for l, k in enumerate(sorted([(i[3], j) for j, i in enumerate(best_scores)], key = lambda x: x[0], reverse=True))}



            #st.write(ranked_acc)
            #st.write(ranked_pre)
            #st.write(ranked_rec)
            #st.write(ranked_f1s)



            mean_avg_acc = np.average([i[0] for i in best_scores])
            mean_avg_pre = np.average([i[1] for i in best_scores])
            mean_avg_rec = np.average([i[2] for i in best_scores])
            mean_avg_f1s = np.average([i[3] for i in best_scores])

            #st.write(mean_avg_acc)
            #st.write(mean_avg_pre)
            #st.write(mean_avg_rec)
            #st.write(mean_avg_f1s)



            if model_type == 'Logistic Regression':
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("### Logistic Regression ###")
                #classifier = LogisticRegression(random_state=0)
                #classifier.fit(x_train, y_train)
                st.markdown("<br>",unsafe_allow_html=True)  
                st.markdown("Summary output")
                st.markdown("<br>",unsafe_allow_html=True) 
                results = smf.logit(f"{churn} ~ {' + '.join([i for i in model_by_Anova.keys()])}", data=data).fit()
                st.write(results.summary())
                st.markdown("<br>",unsafe_allow_html=True)  
                st.markdown("<br>",unsafe_allow_html=True)  
                col1, col2, col3 = st.columns(3, gap = "medium")  
                with col1:
                    st.markdown("Confusion Matrix")
                    #fig = plt.figure()                
                    #cm_model = c.get_cm(data, x, x_train, x_test, y_train, y_test, classifier)
                    #metrics.ConfusionMatrixDisplay(confusion_matrix = cm_model[0], display_labels = [False, True]).plot()
                    metrics.ConfusionMatrixDisplay(confusion_matrix = all_models[0][0], display_labels = [False, True]).plot()
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                    st.info("True Positive is an instance where a positive result is correctly predicted",icon="ℹ️")
                    st.info("True Negative is an instance where a negative result is correctly predicted",icon="ℹ️")
                    st.info("False Positive is an instance where a positive result is incorrectly predicted",icon="ℹ️")
                    st.info("False Negative is an instance where a negative result is incorrectly predicted",icon="ℹ️")
                with col2:
                    st.markdown("ROC Curve")
                    #metrics.plot_roc_curve(cm_model[1], x_test, y_test)
                    #metrics.roc_curve(all_models[0][1], x_test, y_test)
                    RocCurveDisplay.from_estimator(all_models[0][1], x_test, y_test)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                    st.info("A Receiver Operating Characteristic (ROC) curve \
                    is a graphical representation of the performance of a binary \
                    classification model at different classification thresholds. \
                    It plots the true positive rate against the false positive rate at various threshold settings", icon="ℹ️")
                    st.info("The area under the ROC curve (AUC) is a metric used for evaluating the performance of a binary classification model. \
                    Greater area under the ROC curve is proportional to the performance of the binary classification model.", icon="ℹ️")
                with col3:
                    st.markdown("Metrics")
                    #all_metrics = c.get_metrics(y_test, cm_model[2]) 
                    #all_metrics = c.get_metrics(y_test, all_models[0][2]) 
                    #st.dataframe(all_metrics)     
                    st.dataframe(model_metrics[0])
                    st.info("Accuracy is the proportion of corrected identified instances over all the instances in the dataset",icon="ℹ️")
                    st.info("Precision is the proportion of correctly identified positives over all instances identified to be positive",icon="ℹ️")
                    st.info("Recall is the proportion of correctly identified positives over all instances that are infact positive",icon="ℹ️")
                    st.info("F1 score is the harmonic mean between accuracy and precision",icon="ℹ️")
                all_metrics = [i for i in model_metrics[0]["Scores"]]
                #st.write(all_metrics)
                #st.write(mean_avg_acc)
                st.markdown("<br>",unsafe_allow_html=True)
                st.info("Model's metric rank compared to other models", icon="ℹ️")
                #st.markdown("*Note: The number under metric name is the rank order meaning 1 is the best model by certain metric and below is how much it differs from the mean of the metric on all models*")

                col1, col2 = st.columns(2, gap = "medium")  
                with col1:
                    st.metric(label="Accuracy", value = ranked_acc[0] + 1)#, delta=f"{all_metrics[0] - mean_avg_acc }")
                    st.metric(label="Precision", value = ranked_pre[0] + 1)#, delta=f"{all_metrics[1] - mean_avg_pre }")
                with col2:
                    st.metric(label="Recall", value = ranked_rec[0] + 1)#, delta=f"{all_metrics[2] - mean_avg_rec}")
                    st.metric(label="f1-score", value = ranked_f1s[0] + 1)#, delta=f"{all_metrics[3] - mean_avg_f1s }")
                st.markdown("<br>",unsafe_allow_html=True)

            elif model_type == 'Naive Bayes':
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("### Naive Bayes Classifier ###")
                st.markdown("<br>",unsafe_allow_html=True)  
                #classifier = GaussianNB()
                #classifier.fit(x_train, y_train)
                col1, col2, col3 = st.columns(3, gap = "medium")
                with col1:
                    st.markdown("Confusion Matrix")
                    #fig = plt.figure()                
                    #cm_model = c.get_cm(data, x, x_train, x_test, y_train, y_test, classifier)
                    metrics.ConfusionMatrixDisplay(confusion_matrix = all_models[1][0], display_labels = [False, True]).plot()
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                    st.info("True Positive is an instance where a positive result is correctly predicted",icon="ℹ️")
                    st.info("True Negative is an instance where a negative result is correctly predicted",icon="ℹ️")
                    st.info("False Positive is an instance where a positive result is incorrectly predicted",icon="ℹ️")
                    st.info("False Negative is an instance where a negative result is incorrectly predicted",icon="ℹ️")
                with col2:
                    st.markdown("ROC Curve")
                    #metrics.plot_roc_curve(cm_model[1], x_test, y_test)
                    #metrics.plot_roc_curve(all_models[1][1], x_test, y_test)
                    RocCurveDisplay.from_estimator(all_models[1][1], x_test, y_test)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                    st.info("A Receiver Operating Characteristic (ROC) curve \
                    is a graphical representation of the performance of a binary \
                    classification model at different classification thresholds. \
                    It plots the true positive rate against the false positive rate at various threshold settings", icon="ℹ️")
                    st.info("The area under the ROC curve (AUC) is a metric used for evaluating the performance of a binary classification model. \
                    Greater area under the ROC curve is proportional to the performance of the binary classification model.", icon="ℹ️")
                with col3:
                    st.markdown("Metrics")
                    #all_metrics = c.get_metrics(y_test, cm_model[2]) 
                    #all_metrics = c.get_metrics(y_test, all_models[1][2]) 
                    st.dataframe(model_metrics[1])  
                    st.info("Accuracy is the proportion of corrected identified instances over all the instances in the dataset",icon="ℹ️")
                    st.info("Precision is the proportion of correctly identified positives over all instances identified to be positive",icon="ℹ️")
                    st.info("Recall is the proportion of correctly identified positives over all instances that are infact positive",icon="ℹ️")
                    st.info("F1 score is the harmonic mean between accuracy and precision",icon="ℹ️")

                all_metrics = [i for i in model_metrics[1]["Scores"]]
                st.markdown("<br>",unsafe_allow_html=True)
                st.info("Model's metric rank compared to other models", icon="ℹ️")
                #st.markdown("*Note: The number under metric name is the rank order meaning 1 is the best model by certain metric and below is how much it differs from the mean of the metric on all models*")

                col1, col2 = st.columns(2, gap = "medium")  
                with col1:
                    st.metric(label="Accuracy", value = ranked_acc[1] + 1)#, delta=f"{all_metrics[0] - mean_avg_acc}")
                    st.metric(label="Precision", value = ranked_pre[1] + 1)#, delta=f"{all_metrics[1] - mean_avg_pre}")
                with col2:
                    st.metric(label="Recall", value = ranked_rec[1] + 1)#, delta=f"{all_metrics[2] - mean_avg_rec}")
                    st.metric(label="f1-score", value = ranked_f1s[1] + 1)#, delta=f"{all_metrics[3] - mean_avg_f1s}")
                st.markdown("<br>",unsafe_allow_html=True)


            elif model_type == 'Decision Tree':
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("### Decision Tree ###")
                st.markdown("<br>",unsafe_allow_html=True)  
                classifier = tree.DecisionTreeClassifier()
                classifier.fit(x_train, y_train)
                col1, col2, col3 = st.columns(3, gap = "medium")
                with col1:
                    st.markdown("Confusion Matrix")
                    #fig = plt.figure()                
                    #cm_model = c.get_cm(data, x, x_train, x_test, y_train, y_test, classifier)
                    metrics.ConfusionMatrixDisplay(confusion_matrix = all_models[2][0], display_labels = [False, True]).plot()
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                    st.info("True Positive is an instance where a positive result is correctly predicted",icon="ℹ️")
                    st.info("True Negative is an instance where a negative result is correctly predicted",icon="ℹ️")
                    st.info("False Positive is an instance where a positive result is incorrectly predicted",icon="ℹ️")
                    st.info("False Negative is an instance where a negative result is incorrectly predicted",icon="ℹ️")
                with col2:
                    st.markdown("ROC Curve")
                    # metrics.plot_roc_curve(cm_model[1], x_test, y_test)
                    RocCurveDisplay.from_estimator(all_models[2][1], x_test, y_test)
                    #metrics.plot_roc_curve(all_models[2][1], x_test, y_test)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                    st.info("A Receiver Operating Characteristic (ROC) curve \
                    is a graphical representation of the performance of a binary \
                    classification model at different classification thresholds. \
                    It plots the true positive rate against the false positive rate at various threshold settings", icon="ℹ️")
                    st.info("The area under the ROC curve (AUC) is a metric used for evaluating the performance of a binary classification model. \
                    Greater area under the ROC curve is proportional to the performance of the binary classification model.", icon="ℹ️")
                with col3:
                    st.markdown("Metrics")
                    #all_metrics = c.get_metrics(y_test, cm_model[2]) 
                    #all_metrics = c.get_metrics(y_test, all_models[2][2]) 
                    st.dataframe(model_metrics[2])  
                    st.info("Accuracy is the proportion of corrected identified instances over all the instances in the dataset",icon="ℹ️")
                    st.info("Precision is the proportion of correctly identified positives over all instances identified to be positive",icon="ℹ️")
                    st.info("Recall is the proportion of correctly identified positives over all instances that are infact positive",icon="ℹ️")
                    st.info("F1 score is the harmonic mean between accuracy and precision",icon="ℹ️")
                

                all_metrics = [i for i in model_metrics[2]["Scores"]]
                st.markdown("<br>",unsafe_allow_html=True)
                st.info("Model's metric rank compared to other models", icon="ℹ️")
                #st.markdown("*Note: The number under metric name is the rank order meaning 1 is the best model by certain metric and below is how much it differs from the mean of the metric on all models*")

                col1, col2 = st.columns(2, gap = "medium")  
                with col1:
                    st.metric(label="Accuracy", value = ranked_acc[2] + 1)
                    #st.metric(label="Accuracy", value = ranked_acc[2] + 1, delta=f"{all_metrics[0] - mean_avg_acc}")
                    st.metric(label="Precision", value = ranked_pre[2] + 1)#, delta=f"{all_metrics[1] - mean_avg_pre}")
                with col2:
                    st.metric(label="Recall", value = ranked_rec[2] + 1)#, delta=f"{all_metrics[2] - mean_avg_rec}")
                    st.metric(label="f1-score", value = ranked_f1s[2] + 1)#, delta=f"{all_metrics[3] - mean_avg_f1s}")
                st.markdown("<br>",unsafe_allow_html=True)
            else:
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("### Support Vector Machine ###")
                st.markdown("<br>",unsafe_allow_html=True)  
                classifier = svm.SVC()
                classifier.fit(x_train, y_train)
                col1, col2, col3 = st.columns(3, gap = "medium")
                with col1:
                    st.markdown("Confusion Matrix")
                    #fig = plt.figure()                
                    #cm_model = c.get_cm(data, x, x_train, x_test, y_train, y_test, classifier)
                    metrics.ConfusionMatrixDisplay(confusion_matrix = all_models[3][0], display_labels = [False, True]).plot()
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                    st.info("True Positive is an instance where a positive result is correctly predicted",icon="ℹ️")
                    st.info("True Negative is an instance where a negative result is correctly predicted",icon="ℹ️")
                    st.info("False Positive is an instance where a positive result is incorrectly predicted",icon="ℹ️")
                    st.info("False Negative is an instance where a negative result is incorrectly predicted",icon="ℹ️")
                with col2:
                    st.markdown("ROC Curve")
                    RocCurveDisplay.from_estimator(all_models[3][1], x_test, y_test)
                    # metrics.plot_roc_curve(all_models[3][1], x_test, y_test)
                    #metrics.plot_roc_curve(cm_model[1], x_test, y_test)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                    st.info("A Receiver Operating Characteristic (ROC) curve \
                    is a graphical representation of the performance of a binary \
                    classification model at different classification thresholds. \
                    It plots the true positive rate against the false positive rate at various threshold settings", icon="ℹ️")
                    st.info("The area under the ROC curve (AUC) is a metric used for evaluating the performance of a binary classification model. \
                    Greater area under the ROC curve is proportional to the performance of the binary classification model.", icon="ℹ️")
                with col3:
                    st.markdown("Metrics")
                    #all_metrics = c.get_metrics(y_test, all_models[3][2]) 
                    # all_metrics = c.get_metrics(y_test, cm_model[2]) 
                    st.dataframe(model_metrics[3])      
                    st.info("Accuracy is the proportion of corrected identified instances over all the instances in the dataset",icon="ℹ️")
                    st.info("Precision is the proportion of correctly identified positives over all instances identified to be positive",icon="ℹ️")
                    st.info("Recall is the proportion of correctly identified positives over all instances that are infact positive",icon="ℹ️")
                    st.info("F1 score is the harmonic mean between accuracy and precision",icon="ℹ️")
                
                st.markdown("<br>",unsafe_allow_html=True)
                all_metrics = [i for i in model_metrics[3]["Scores"]]
                st.info("Model's metric rank compared to other models", icon="ℹ️")
               # st.markdown("*Note: The number under metric name is the rank order meaning 1 is the best model by certain metric and below is how much it differs from the mean of the metric on all models*")

                col1, col2 = st.columns(2, gap = "medium")  
                with col1:
                    st.metric(label="Accuracy", value = ranked_acc[3] + 1)#, delta=f"{all_metrics[0] - mean_avg_acc}")
                    st.metric(label="Precision", value = ranked_pre[3] + 1)#, delta=f"{all_metrics[1] - mean_avg_pre}")
                with col2:
                    st.metric(label="Recall", value = ranked_rec[3] + 1)#, delta=f"{all_metrics[2] - mean_avg_rec}")
                    st.metric(label="f1-score", value = ranked_f1s[3] + 1)#, delta=f"{all_metrics[3] - mean_avg_f1s}")
                st.markdown("<br>",unsafe_allow_html=True)

            model_scores = [(model["Scores"].sum(), index) for index, model in enumerate(model_metrics)]
            best_model_i = sorted(model_scores, key = lambda x: x[0],reverse = True)[0][1]
            st.info(f"{model_names[best_model_i]} is the model with the best model metrics by our parameters", icon="ℹ️")
            st.info(f"We have selected the model with the highest summation of all metrics since all metrics have the property that a higher metric is proportional to how well the model perform", icon="ℹ️")
            st.markdown("<br>",unsafe_allow_html=True)

            # YOU CAN SELECT THE ML MODEL YOU WANT TO USE RECOMMENDED IS THE EBST model
            st.markdown(f"### Optional: Specify a machine learning model to use")  
            st.markdown("<br>",unsafe_allow_html=True)
            best_m = model_names[best_model_i]
            model_names.remove(best_m)
            model_names.insert(0, best_m)
            selected_model = st.radio(f"By default we have selected {best_m} (the best model by our parameters)", model_names)
            st.markdown("<br>",unsafe_allow_html=True)

            ## TO FIX
            ## MAKE SURE THAT YOU CAN CHANGE THE MODEL SELECTED
        with tab5:
            st.markdown(f"# Investigation of model")
            st.markdown("<br>",unsafe_allow_html=True)
            st.info(f"We use the selected model to make probability predictions of churn for each customer and using this, we can glean stronger insights between churn and the variables", icon="ℹ️") 

            classifier = all_classifiers[best_model_i]
            predict_prob = classifier.predict_proba(list(model_by_Anova.keys()))
                                    
            #st.dataframe(data.iloc[most_riskoc.index(max(most_riskoc))])
            st.markdown("<br>",unsafe_allow_html=True)  
            cola, colb = st.columns(2, gap = "large")
            with cola:
                st.markdown(f"### How churn differs between two variables")
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("Select two variables for the plot")
                col1, col2 = st.columns(2, gap = "large")
                with col1:
                    x_axis = st.radio("X_axis", model_by_Anova)
                with col2:
                    y_axis = st.radio("Y_axis", model_by_Anova)
                st.markdown("<br>",unsafe_allow_html=True)
                fig = plt.figure()
                sns.set_theme(color_codes=True)
                #plot1 = sns.barplot(x="Dependents", y="Partner",  hue="Churn", data=vis_data)
                plot1 = sns.lmplot(x=x_axis, y=y_axis, markers=["o", "x"], hue=churn, data=not_encoded)
                plt.title(f"{y_axis} vs {x_axis}")
                st.pyplot(plot1)
                st.info(f"The following scatterplot allows us to observe a relationship between {x_axis} and {y_axis}. Points of intersection between the not churn and churn line would indicate beyond a certain x and y, customers would begin to churn.", icon="ℹ️")

            with colb:
                #st.markdown(f"### Probability of churn amongst all customers")
                #st.markdown("<br>",unsafe_allow_html=True)
                #st.markdown(f"Probability of churn")
                #fig = plt.figure()
                #hist = pd.DataFrame(predict_prob)[1]
                #st.bar_chart(pd.DataFrame(hist.value_counts())) 
                #st.bar_chart(predict_prob)
                st.markdown(f"### Average Churn Rate of unique values in a variable")
                st.markdown("<br>",unsafe_allow_html=True)
                #st.markdown(f"Select a variable to view")
                x_axis = st.radio("Select a variable to view:", model_by_Anova)
                numerical = False
                if x_axis not in variable_data_type:
                    pass
                elif variable_data_type[x_axis] == "Numerical":
                    subset =  st.slider("Select how much to subset the numerical values", 0, 20, 4)
                    numerical = subset
                st.markdown("<br>",unsafe_allow_html=True)
                descending_churn = c.top_churn(predict_prob, all_customer_ID, not_scaled, customerID)
                churn_by_var = c.avg_churn(descending_churn, x_axis, numerical)
                st.bar_chart(churn_by_var)
                st.info(f"The following barplot shows the different churn rate for each variable groups.", icon="ℹ️")


            


        with tab6:
            st.markdown(f"# Making predictions")
            st.markdown("<br>",unsafe_allow_html=True)
            st.info("Using the machine learning model we built previously, input a dataset with the same columns as the previously used dataset other than churn to predict churn. ",icon="ℹ️")
            new_data = st.file_uploader(label="Upload new dataset")
            #st.write(best_model_i)
            if new_data is not None:
                try:
                    new_data = pd.DataFrame(pd.read_csv(new_data, encoding="unicode_escape"))
                    copy_new = new_data
                except:
                    st.error("Error: Can only receive inputs of .csv type",icon="🛑")
                    quit()
                st.info(f"Your input", icon="ℹ️") 
                st.dataframe(new_data)
                new_cust_ID = new_data[customerID]
                new_data = new_data[[i for i in variables if i != churn]]
                new_data_info = c.clean_data(new_data, False, vif_thres_test, propo_thres_test, dispro_thres_test, new_cust_ID, variable_data_type, customerID, sd_threshold, ordinal_rank, churn)
                new_data = new_data_info[0]
                not_scaled = new_data_info[2]
                churn_col = classifier.predict(new_data[model_by_Anova])
                new_predict_prob = classifier.predict_proba(new_data[list(model_by_Anova.keys())] )
                new_data[churn] = churn_col
                new_most_riskoc = sorted([(i[1], index) for index, i in enumerate(new_predict_prob)], key = lambda x: x[0], reverse=True)
                new_data["Churn Rate"] = [i[0] for i in new_most_riskoc]
                index_arrangement = [i[1] for i in new_most_riskoc]
                #descending_churn = original_data.iloc[index_arrangement]
                descending_churn = pd.DataFrame()
                descending_churn[customerID] = [all_customer_ID[i] for i in index_arrangement]
                new_data[customerID] = descending_churn[customerID]
                # st.write(new_data.columns)
                #st.write(f"{cols} is columns")
                new_data = new_data[[customerID, churn, "Churn Rate"]]
                copy_new = pd.merge(copy_new, new_data, on=customerID, how='inner')


                
                #new_data["Churn Rate"] = new_predict_prob
                st.info(f"Dataset with an additional churn and predicted churn columns", icon="ℹ️") 
                st.dataframe(copy_new)
                st.markdown("<br>",unsafe_allow_html=True)  
                st.markdown("<br>",unsafe_allow_html=True)
                #fig = plt.figure()
                #plt.title(f"Churn")
                #new_data["Churn"].value_counts().plot(kind='bar')
                #st.pyplot(fig)   
                st.info(f"Distribution of churn on the predicted churn", icon="ℹ️") 

                st.markdown("<br>",unsafe_allow_html=True)   

                #st.dataframe(new_data["Churn"])
                st.bar_chart(new_data[churn].value_counts())

                #st.bar_chart(pd.DataFrame(new_data["Churn"])) 
                st.markdown("<br>",unsafe_allow_html=True)                
                st.download_button(
                    label="Download new data as .csv",
                    data=new_data.to_csv(index=False).encode('utf-8'),
                    file_name='added_churn.csv',
                    mime='text/csv',
                )

                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("### Customers with highest risk of Churn")
                st.markdown("<br>",unsafe_allow_html=True)

                percent_view = st.slider("Choose to view the top specified percentage of customers", 0, 100, 100)
                st.markdown("<br>",unsafe_allow_html=True)

                #most_riskoc = sorted([(i[1], index) for index, i in enumerate(predict_prob)], key = lambda x: x[0], reverse=True)
                #index_arrangement = [i[1] for i in most_riskoc]
                #descending_churn = original_data.iloc[index_arrangement]
                #descending_churn = pd.DataFrame()
                #descending_churn["customerID"] = [all_customer_ID[i] for i in index_arrangement]
                #descending_churn["Churn Rate"] = [i[0] for i in most_riskoc]
                #descending_churn = pd.merge(descending_churn, original_data, on='customerID', how='inner')
                descending_churn = c.top_churn(predict_prob, new_cust_ID, not_scaled, customerID)
                number_rows = len(descending_churn)
                st.dataframe(descending_churn.head(int(percent_view / 100 * (number_rows - 1))))
                st.markdown("<br>",unsafe_allow_html=True)

                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("### Customers by risk category")
                st.markdown("<br>",unsafe_allow_html=True)
                low = st.number_input('Choose the churn rate for low risk customers', value=0.4)
                high = st.number_input('Choose the churn rate for high risk customers', value=0.5)
                if low < 0:
                    st.error("Churn rate must be positive",icon="🛑")
                    quit()
                elif low == 0:
                    st.error("Churn rate must be non-zero",icon="🛑")
                    quit()
                elif low >= 1 or high >= 1:
                    st.error("Churn rate must be between 0 and less than 1",icon="🛑")
                    quit()
                elif low == high:
                    st.error("Upper bound for low risk customers must be strictly greater than lower bound for high risk customers",icon="🛑")
                    quit()
                low_risk = descending_churn[descending_churn['Churn Rate'] < low]
                medium_risk = descending_churn[(descending_churn['Churn Rate'] >= low) & (descending_churn['Churn Rate'] < high)]
                high_risk = descending_churn[descending_churn['Churn Rate'] >= high]

                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("#### The low risk customers")
                st.info(f"Customers that have a churn rate that is less than {low}", icon="ℹ️")
                st.dataframe(low_risk)
                st.download_button(
                    label="Download low risk customers as .csv",
                    data=low_risk.to_csv(index=False).encode('utf-8'),
                    file_name='low_risk_customers.csv',
                    mime='text/csv',
                )
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("#### The medium risk customers")
                st.info(f"Customers that have a churn rate that is greater than {low} but less than {high}", icon="ℹ️")
                st.dataframe(medium_risk)
                st.download_button(
                    label="Download medium risk customers as .csv",
                    data=medium_risk.to_csv(index=False).encode('utf-8'),
                    file_name='medium_risk_customers.csv',
                    mime='text/csv',
                )
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("#### The high risk customers")
                st.info(f"Customers that have a churn rate that is greater than {high}", icon="ℹ️")
                st.dataframe(high_risk)
                st.download_button(
                    label="Download high risk customers as .csv",
                    data=high_risk.to_csv(index=False).encode('utf-8'),
                    file_name='top_risk_customers.csv',
                    mime='text/csv',
                )

                                

            #st.dataframe(model_metrics[best_model_i])   
            # 
            # CHANGE HERE   
            #best_scores = [i for i in model_metrics[best_model_i]["Scores"]]


            #st.write(best_scores) 
            
        #else:
            #st.error("Fill in all fields")
#elif data is not None and is_churn_col:
 #   st.warning("Error: No churn column")
else:
    st.info("No input file")


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
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
import churn_functions as c
from sklearn import tree
from sklearn import svm
import statsmodels.formula.api as smf
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import RocCurveDisplay
import random 
from bokeh.plotting import figure


st.markdown("""<h1>Preprocessing</h1>""", unsafe_allow_html=True)  
st.markdown("<br>",unsafe_allow_html=True)
data = st.file_uploader(label="Upload dataset")
if data is not None:
    try:
        data = pd.DataFrame(pd.read_csv(data, encoding="unicode_escape"))
    except:
        st.error("Can only receive inputs of .csv type")
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


if data is not None and "Churn" in data.columns and "customerID" in data.columns:
    if len(data.columns) <= 2:
        st.warning("Data contains insufficient columns")
    else:
        original_data = data
        all_customer_ID = data["customerID"]
        st.markdown("Your input")
        st.dataframe(data)

        st.markdown("Select the columns you want to use")
        submitted = select_columns(data)
        data.drop(variables, axis=1)
        if len(variables) > 0:
            st.write(variables)
            st.markdown("Specify the type of data each variable is")
            submitted = var_data_types(data)
            st.write(variable_data_type)

            if len(variable_data_type) == len(variables):
                

                # READ THIS !!!!!
                #     df["Scale"] = df["Score"].replace(scale_mapper)

                # Ordinal variables 
                st.markdown("For all ordinal variables, select the hierachical order of values")
                ordinal_vars = [(key, data[key].unique()) for key, value in variable_data_type.items() if value == "Ordinal"]
                ordinal_rank = {}
                with st.form("ordinal"):
                    for vars in ordinal_vars:
                        st.markdown(vars[0])
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
                    st.write(check_order)
                    if len(set(check_order)) != len(check_order):
                        default_order.append(var)

                st.write(default_order)
                if len(default_order) > 0:
                    st.warning("You have duplicate orders. Ordinal scale is reset to the default encoding")
                    for default_ord in default_order:
                        orders = ordinal_rank[default_ord]
                        i = 0
                        for pos in orders:
                            orders[pos] = i
                            i += 1
                        


                st.write(ordinal_rank)
                
                st.markdown("Specify the regex format or a list of possible values seperated by commas for each variable (*Optional*)")
                variables.append("Churn")
                submitted = var_format(data, variable_data_type)
                st.write(variable_format)
                #if len(variable_format) == len(variables):
                data = data[variables]

                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("# EDA (Exploratory Data Analysis)")
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("### Distribution of variables ###")
                st.markdown("<br>",unsafe_allow_html=True)
                st.write(data.describe(include = 'all'))
                st.markdown("<br>",unsafe_allow_html=True)
                        

                st.markdown("### Barplot for selected variable ###")
                st.markdown("<br>",unsafe_allow_html=True)
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
                st.markdown("Select two variables for the plot")
                col1, col2 = st.columns(2, gap = "large")
                with col1:
                    x_scplt = st.radio("X axis", variables)
                with col2:
                    y_scplt = st.radio("Y axis", variables)

                fig = plt.figure()
                plt.scatter(data[x_scplt], data[y_scplt])
                plt.xlabel(x_scplt)
                plt.ylabel(y_scplt)
                plt.title(f"{y_scplt} vs {x_scplt}")
                st.pyplot(fig)
                


             
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("### Cleaning the data")
                st.markdown("<br>",unsafe_allow_html=True)
                vif_thres_test = st.slider("Select the VIF threshold", 0, 100, 10)
                propo_thres_test = st.slider("Select the proportion of nulls threshold", 0.0, 1.0, 0.5)
                dispro_thres_test = st.slider("Select the maximum disproportion of variables within a column", 0.0, 1.0, 0.9)
                st.markdown("<br>",unsafe_allow_html=True)


                data_vif = c.clean_data(data, True, vif_thres_test, propo_thres_test, dispro_thres_test, all_customer_ID, variable_data_type)
                data = data_vif[1]
                to_remove, to_purge,to_drop, curr_customer_ID = data_vif[2], data_vif[3], data_vif[4], data_vif[5]
                not_scaled = data_vif[6]

                try:
                    st.write(f"Column/s that exceed the given proportion of nulls {propo_thres_test} is/are {' '.split(to_remove)}")
                except:
                    st.write("No column has excessive null entries by given threshold")

                try:
                    st.write(f"Column/s that have a strongly disproportionate distribution of variable at {dispro_thres_test} is/are {' '.split(to_purge)}")
                except:
                    st.write(f"No column has have a strongly disproportionate distribution of variable at {dispro_thres_test}")
        
                try:
                    st.write(f"Variable/s which exceed the given threshold VIF of {vif_thres_test} is/are {' '.split(to_drop)}")
                except:
                    st.write(f"No variables exceed the given threshold VIF of {vif_thres_test}")
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
                with col2:
                    st.markdown("VIF of each variable")
                    #vif_df = c.vif_df(data)
                    st.dataframe(data_vif[0])
                st.markdown("<br>",unsafe_allow_html=True)

                st.markdown("# Splitting the Data #")
                st.markdown("<br>",unsafe_allow_html=True)
                num_cols = len(data.columns)
                x = data.iloc[:, :-1]
                y = data.iloc[:, num_cols - 1]

                st_x = StandardScaler()
                x_scaled = st_x.fit_transform(x)
                x = pd.DataFrame(x_scaled, index=x.index, columns=x.columns)

                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
                topcol1, topcol2 = st.columns(2, gap = "large")
                with topcol1:
                    st.markdown("### Training set ###")
                    col1, col2 = st.columns(2, gap = "large")
                    with col1:
                        st.markdown("X-train")
                        st.dataframe(x_train)
                    with col2:
                        st.markdown("X-test")
                        st.dataframe(x_test)
                with topcol2:
                    st.markdown("### Test set ###")
                    col1, col2 = st.columns(2, gap = "large")
                    with col1:
                        st.markdown("Y-train")
                        st.dataframe(y_train)
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
                predictors = data.columns[:-1]
                selected_pred = [i for i in model_by_Anova]

                if len(selected_pred) == len(predictors):
                    st.markdown(f"Using the F-statistic, the variables that yields a p value of less than 0.05 are {', '.join(selected_pred)}")
                else:
                    st.markdown(f"Using the F-statistic, the variables that yields a p value of less than 0.05 are {', '.join(selected_pred)} excluding {' '.join(predictors.difference(selected_pred))}")


                
                x_train = x_train.iloc[:, [i for i in model_by_Anova.values()]]
                x_test = x_test.iloc[:, [i for i in model_by_Anova.values()]]  
                st.markdown("### After Feature Selection ###")


                st.markdown("<br>",unsafe_allow_html=True)
                col1, col2 = st.columns(2, gap = "large")
                with col1:
                    st.markdown("Modified X-train")
                    st.dataframe(x_train)
                with col2:
                    st.markdown("Modified X-test")
                    st.dataframe(x_test)

                st.markdown("# Model Results #")
                st.markdown("<br>",unsafe_allow_html=True)
                model_type = st.radio("Select to view each model", ('Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Support Vector Machine'))

                all_classifiers = [LogisticRegression(random_state=0), GaussianNB(), tree.DecisionTreeClassifier(), svm.SVC()]
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
                    results = smf.logit(f"Churn ~ {' + '.join([i for i in model_by_Anova.keys()])}", data=data).fit()
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
                    with col2:
                        st.markdown("ROC Curve")
                        #metrics.plot_roc_curve(cm_model[1], x_test, y_test)
                        #metrics.roc_curve(all_models[0][1], x_test, y_test)
                        RocCurveDisplay.from_estimator(all_models[0][1], x_test, y_test)
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                    with col3:
                        st.markdown("Metrics")
                        #all_metrics = c.get_metrics(y_test, cm_model[2]) 
                        #all_metrics = c.get_metrics(y_test, all_models[0][2]) 
                        #st.dataframe(all_metrics)     
                        st.dataframe(model_metrics[0])
                        
                    all_metrics = [i for i in model_metrics[0]["Scores"]]
                    #st.write(all_metrics)
                    #st.write(mean_avg_acc)
                    st.markdown("<br>",unsafe_allow_html=True)
                    st.markdown("Model's metric rank compared to other models")
                    st.markdown("*Note: The number under metric name is the rank order meaning 1 is the best model by certain metric and below is how much it differs from the mean of the metric on all models*")

                    col1, col2 = st.columns(2, gap = "medium")  
                    with col1:
                        st.metric(label="Accuracy", value = ranked_acc[0] + 1, delta=f"{all_metrics[0] - mean_avg_acc }")
                        st.metric(label="Precision", value = ranked_pre[0] + 1, delta=f"{all_metrics[1] - mean_avg_pre }")
                    with col2:
                        st.metric(label="Recall", value = ranked_rec[0] + 1, delta=f"{all_metrics[2] - mean_avg_rec}")
                        st.metric(label="f1-score", value = ranked_f1s[0] + 1, delta=f"{all_metrics[3] - mean_avg_f1s }")
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
                    with col2:
                        st.markdown("ROC Curve")
                        #metrics.plot_roc_curve(cm_model[1], x_test, y_test)
                        #metrics.plot_roc_curve(all_models[1][1], x_test, y_test)
                        RocCurveDisplay.from_estimator(all_models[1][1], x_test, y_test)
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                    with col3:
                        st.markdown("Metrics")
                        #all_metrics = c.get_metrics(y_test, cm_model[2]) 
                        #all_metrics = c.get_metrics(y_test, all_models[1][2]) 
                        st.dataframe(model_metrics[1])  

                    all_metrics = [i for i in model_metrics[1]["Scores"]]
                    st.markdown("<br>",unsafe_allow_html=True)
                    st.markdown("Model's metric rank compared to other models")
                    st.markdown("*Note: The number under metric name is the rank order meaning 1 is the best model by certain metric and below is how much it differs from the mean of the metric on all models*")

                    col1, col2 = st.columns(2, gap = "medium")  
                    with col1:
                        st.metric(label="Accuracy", value = ranked_acc[1] + 1, delta=f"{all_metrics[0] - mean_avg_acc}")
                        st.metric(label="Precision", value = ranked_pre[1] + 1, delta=f"{all_metrics[1] - mean_avg_pre}")
                    with col2:
                        st.metric(label="Recall", value = ranked_rec[1] + 1, delta=f"{all_metrics[2] - mean_avg_rec}")
                        st.metric(label="f1-score", value = ranked_f1s[1] + 1, delta=f"{all_metrics[3] - mean_avg_f1s}")
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
                    with col2:
                        st.markdown("ROC Curve")
                       # metrics.plot_roc_curve(cm_model[1], x_test, y_test)
                        RocCurveDisplay.from_estimator(all_models[2][1], x_test, y_test)
                        #metrics.plot_roc_curve(all_models[2][1], x_test, y_test)
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                    with col3:
                        st.markdown("Metrics")
                        #all_metrics = c.get_metrics(y_test, cm_model[2]) 
                        #all_metrics = c.get_metrics(y_test, all_models[2][2]) 
                        st.dataframe(model_metrics[2])  
                    
                    all_metrics = [i for i in model_metrics[2]["Scores"]]
                    st.markdown("<br>",unsafe_allow_html=True)
                    st.markdown("Model's metric rank compared to other models")
                    st.markdown("*Note: The number under metric name is the rank order meaning 1 is the best model by certain metric and below is how much it differs from the mean of the metric on all models*")

                    col1, col2 = st.columns(2, gap = "medium")  
                    with col1:
                        st.metric(label="Accuracy", value = ranked_acc[2] + 1, delta=f"{all_metrics[0] - mean_avg_acc}")
                        st.metric(label="Precision", value = ranked_pre[2] + 1, delta=f"{all_metrics[1] - mean_avg_pre}")
                    with col2:
                        st.metric(label="Recall", value = ranked_rec[2] + 1, delta=f"{all_metrics[2] - mean_avg_rec}")
                        st.metric(label="f1-score", value = ranked_f1s[2] + 1, delta=f"{all_metrics[3] - mean_avg_f1s}")
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
                    with col2:
                        st.markdown("ROC Curve")
                        RocCurveDisplay.from_estimator(all_models[3][1], x_test, y_test)
                       # metrics.plot_roc_curve(all_models[3][1], x_test, y_test)
                        #metrics.plot_roc_curve(cm_model[1], x_test, y_test)
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                    with col3:
                        st.markdown("Metrics")
                        #all_metrics = c.get_metrics(y_test, all_models[3][2]) 
                       # all_metrics = c.get_metrics(y_test, cm_model[2]) 
                        st.dataframe(model_metrics[3])      
                    
                    st.markdown("<br>",unsafe_allow_html=True)
                    all_metrics = [i for i in model_metrics[3]["Scores"]]
                    st.markdown("Model's metric rank compared to other models")
                    st.markdown("*Note: The number under metric name is the rank order meaning 1 is the best model by certain metric and below is how much it differs from the mean of the metric on all models*")

                    col1, col2 = st.columns(2, gap = "medium")  
                    with col1:
                        st.metric(label="Accuracy", value = ranked_acc[3] + 1, delta=f"{all_metrics[0] - mean_avg_acc}")
                        st.metric(label="Precision", value = ranked_pre[3] + 1, delta=f"{all_metrics[1] - mean_avg_pre}")
                    with col2:
                        st.metric(label="Recall", value = ranked_rec[3] + 1, delta=f"{all_metrics[2] - mean_avg_rec}")
                        st.metric(label="f1-score", value = ranked_f1s[3] + 1, delta=f"{all_metrics[3] - mean_avg_f1s}")
                    st.markdown("<br>",unsafe_allow_html=True)

                model_scores = [(model["Scores"].sum(), index) for index, model in enumerate(model_metrics)]
                best_model_i = sorted(model_scores, key = lambda x: x[0],reverse = True)[0][1]
                st.markdown(f"## {model_names[best_model_i]} is the model with the best model metrics")  
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

                st.markdown(f"# Dashboard")
                classifier = all_classifiers[best_model_i]
                predict_prob = classifier.predict_proba(data[model_by_Anova])
                                        
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
                    plot1 = sns.lmplot(x=x_axis, y=y_axis, markers=["o", "x"], hue="Churn", data=data)
                    plt.title(f"{y_axis} vs {x_axis}")
                    st.pyplot(plot1)
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
                    descending_churn = c.top_churn(predict_prob, all_customer_ID, not_scaled)
                    churn_by_var = c.avg_churn(descending_churn, x_axis, numerical)
                    st.bar_chart(churn_by_var)


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
                number_rows = len(descending_churn)
                st.dataframe(descending_churn.head(int(percent_view / 100 * (number_rows - 1))))
                st.markdown("<br>",unsafe_allow_html=True)
                st.download_button(
                    label="Download top risk customers as .csv",
                    data=descending_churn.to_csv(index=False).encode('utf-8'),
                    file_name='top_risk_customers.csv',
                    mime='text/csv',
                )

                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("### Customers by risk category")
                st.markdown("<br>",unsafe_allow_html=True)
                low = st.number_input('Choose the churn rate for low risk customers', min_value = 0.0, max_value = 1.0, value=0.4)
                high = st.number_input('Choose the churn rate for high risk customers', min_value = low, max_value = 1.0, value=0.5)
                low_risk = descending_churn[descending_churn['Churn Rate'] < low]
                medium_risk = descending_churn[(descending_churn['Churn Rate'] > low) & (descending_churn['Churn Rate'] < high)]
                high_risk = descending_churn[descending_churn['Churn Rate'] > high]

                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("#### The low risk customers")
                st.markdown(f"Customers that have a churn rate that is less than {low}")
                st.dataframe(low_risk)
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("#### The medium risk customers")
                st.markdown(f"Customers that have a churn rate that is greater than {low} but less than {high}")
                st.dataframe(medium_risk)
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("#### The high risk customers")
                st.markdown(f"Customers that have a churn rate that is greater than {high}")
                st.markdown("<br>",unsafe_allow_html=True)
                st.dataframe(high_risk)




                st.markdown(f"# Making predictions")
                st.markdown("<br>",unsafe_allow_html=True)
                new_data = st.file_uploader(label="Upload new dataset")
                #st.write(best_model_i)
                if new_data is not None:
                    try:
                        new_data = pd.DataFrame(pd.read_csv(new_data, encoding="unicode_escape"))
                        copy_new = new_data
                    except:
                        st.error("Can only receive inputs of .csv type")
                        quit()
                    st.markdown(f"Input")   
                    st.dataframe(new_data)
                    new_cust_ID = new_data["customerID"]
                    new_data = new_data[[i for i in variables if i != "Churn"]]
                    new_data = c.clean_data(new_data, False, vif_thres_test, propo_thres_test, dispro_thres_test, new_cust_ID, variable_data_type)
                    churn_col = classifier.predict(new_data[model_by_Anova])
                    new_predict_prob = classifier.predict_proba(new_data[model_by_Anova])
                    new_data["Churn"] = churn_col
                    new_most_riskoc = sorted([(i[1], index) for index, i in enumerate(new_predict_prob)], key = lambda x: x[0], reverse=True)
                    new_data["Churn Rate"] = [i[0] for i in new_most_riskoc]
                    index_arrangement = [i[1] for i in new_most_riskoc]
                    #descending_churn = original_data.iloc[index_arrangement]
                    descending_churn = pd.DataFrame()
                    descending_churn["customerID"] = [all_customer_ID[i] for i in index_arrangement]
                    new_data["customerID"] = descending_churn["customerID"]
                   # st.write(new_data.columns)
                    #st.write(f"{cols} is columns")
                    new_data = new_data[["customerID", "Churn", "Churn Rate"]]
                    copy_new = pd.merge(copy_new, new_data, on='customerID', how='inner')


                    
                    #new_data["Churn Rate"] = new_predict_prob
                    st.markdown(f"Output")   
                    st.dataframe(copy_new)
                    st.markdown("<br>",unsafe_allow_html=True)  
                    st.markdown("<br>",unsafe_allow_html=True)
                    #fig = plt.figure()
                    #plt.title(f"Churn")
                    #new_data["Churn"].value_counts().plot(kind='bar')
                    #st.pyplot(fig)   
                    st.markdown(f"Churn")  
                    st.markdown("<br>",unsafe_allow_html=True)   

                    #st.dataframe(new_data["Churn"])
                    st.bar_chart(new_data["Churn"].value_counts())

                    #st.bar_chart(pd.DataFrame(new_data["Churn"])) 
                    st.markdown("<br>",unsafe_allow_html=True)                
                    st.download_button(
                        label="Download new data as .csv",
                        data=new_data.to_csv(index=False).encode('utf-8'),
                        file_name='added_churn.csv',
                        mime='text/csv',
                    )
                                    

                #st.dataframe(model_metrics[best_model_i])   
                # 
                # CHANGE HERE   
                #best_scores = [i for i in model_metrics[best_model_i]["Scores"]]
        

                #st.write(best_scores) 
                
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

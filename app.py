from dash import Dash, html, dcc, callback, Input, Output
#import simulation as sim
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split    

np.random.seed(43)

def roc_curve(y_test, preds):
    thresholds = np.linspace(0, 1 , 100)
    tpr=[]
    fpr=[]
    for threshold in thresholds:
        TP = len([1 for i in range(len(y_test)) if (y_test[i]==0 and preds[i]>threshold)])
        FP = len([1 for i in range(len(y_test)) if (y_test[i]==1 and preds[i]>threshold)])
        TN = len([1 for i in range(len(y_test)) if (y_test[i]==1 and preds[i]<=threshold)])
        FN = len([1 for i in range(len(y_test)) if (y_test[i]==0 and preds[i]<=threshold)])
        tpr_val = TP/(TP+FN)
        fpr_val = FP/(FP+TN)
        tpr.append(tpr_val)
        fpr.append(fpr_val)
    return tpr, fpr, thresholds

app = Dash(__name__)

app.layout = html.Div([
    html.Div(children='Understanding ROC and AUC', style={'textAlign': 'center', 'color': '#242a44', 'fontSize': 30}),
    html.Hr(),
    html.Div([
        html.Div(children='Mean for the positive data samples'),
        dcc.Slider(1, 10, value=1, id='mean_pos')], style={'float':'left', 'width':'33%'}),
    html.Div([
        html.Div(children='Mean for the negative data samples'),
        dcc.Slider(1, 10, value=1, id='mean_neg')], style={'float':'left', 'width':'33%'}),
    html.Div([
        html.Div(children='standard deviation for the positive data samples'),
        dcc.Slider(1, 10, value=1, id='std_pos')], style={'float':'left', 'width':'33%'}),
    html.Div([
        html.Div(children='standard deviation for the negative data samples'),
        dcc.Slider(1, 10, value=1, id='std_neg')], style={'float':'left', 'width':'33%'}),
    html.Div([
        html.Div(children='Number of negative data samples'),
        dcc.Slider(2, 1000, value=100, id='no_neg')], style={'float':'left', 'width':'33%'}),
    html.Div([
        html.Div(children='Number of positive data samples'),
        dcc.Slider(2, 1000, value=100, id='no_pos')], style={'float':'left', 'width':'33%'}),

    dcc.Graph(figure={}, id='distribution_graph', style={'float':'left', 'width':'50%', 'text-align':'center'}),
    dcc.Graph(figure={}, id='ROC_graph_own', style={'float':'left', 'width':'50%', 'text-align':'center'}),

    html.Div([
        html.Div(children = 'Notes', style={'textAlign': 'center', 'color': '#242a44', 'fontSize': 20}),
        html.Hr(),
        html.Div(children = ' - A diagonal line(line connecting (0,0) and (1,1)) in ROC curve represents random performance, a line below it is worse than random prediction and line above it is better than random prediction.'),
        html.Div(children = ' - Greater the area under curve the better the model is.'),
        html.Div(children = ' - In the above case Gaussian Naive Bayes model is better than logistic regression model, as you can observe with varying AUC, as you toggle different values.'),
        html.Div(children = ' - This metrics should be used in case of imbalance in the dataset, for model comparison, while hyperparameter tuning(as it evaluates at different threshold settings) and for visual representation.'),
        html.Div(children = ' - ROC curve evaluates the model for different threshold settings, for a given model, that is each point is a tpr and fpr for each of the threshold value selected.'),
        html.Div(children = ' - This is only for binary classification, for multiclass classification, constructing ROC becomes complicated.')])
])

@callback(
    Output(component_id='distribution_graph', component_property='figure'),
    Output(component_id='ROC_graph_own', component_property='figure'),
    Input(component_id='mean_pos', component_property='value'),
    Input(component_id='mean_neg', component_property='value'),
    Input(component_id='std_pos', component_property='value'),
    Input(component_id='std_neg', component_property='value'),
    Input(component_id='no_pos', component_property='value'),
    Input(component_id='no_neg', component_property='value'),
    #Input(component_id='threshold', component_property='value')
)
def distibution_graph(mean_pos, mean_neg, std_pos, std_neg, no_pos, no_neg):
    
    positive_datasamples = np.random.normal(loc=mean_pos, scale=std_pos, size=no_pos)
    negative_datasamples = np.random.normal(loc=mean_neg, scale=std_neg, size=no_neg)
    X_all = np.concatenate((positive_datasamples, negative_datasamples), axis=0)
    y_all = np.concatenate((np.zeros((len(positive_datasamples), 1)),np.ones((len(negative_datasamples), 1)) ), axis=0)
    X_all, y_all = shuffle(X_all, y_all)
    X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.33, random_state=42)
    X = X.reshape((-1,1))
    X_test = X_test.reshape((-1,1))
    min_val = min(min(positive_datasamples), min(negative_datasamples))
    max_val = max(max(positive_datasamples), max(negative_datasamples))
    fig1 = ff.create_distplot([positive_datasamples, negative_datasamples] , ['positive samples', 'negative samples'], show_hist=False, show_rug=False)
    fig1.update_layout(title='Dataset Distribution', title_x=0.5)
    model1 = LogisticRegression(random_state=0).fit(X, y)
    probs1 = model1.predict_proba(X_test)
    preds1 = probs1[:,1]

    model2 = GaussianNB().fit(X, y)
    probs2 = model2.predict_proba(X_test)
    preds2 = probs2[:,1]
    fpr1, tpr1, _ = roc_curve(y_test, preds1)
    fpr2, tpr2, _ = roc_curve(y_test, preds2)
    
    trace1 = go.Scatter(x=fpr1, y=tpr1, mode='lines', name='Logistic Regression model')
    trace2 = go.Scatter(x=fpr2, y=tpr2, mode='lines', name='Gaussian Naive Bayes')
    layout = go.Layout(
    title='ROC Curves',
    xaxis=dict(title='False Positive Rate'),
    yaxis=dict(title='True Positive Rate'),
    showlegend=True
)
    fig2 = go.Figure(data=[trace1, trace2], layout=layout)

    fig2.update_layout(title_x=0.5)
    return fig1, fig2

if __name__ == '__main__':
    app.run(debug=True)
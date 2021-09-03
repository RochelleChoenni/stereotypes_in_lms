
import json
import plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA

def pca_scatterplot_3D(Model, Category, sample=20):
    names = {'BERT-B': 'bert-base-uncased', 'BERT-L': 'bert-large-uncased', 'RoBERTa-B': 'roberta-base', 'RoBERTa-L': 'roberta-large',
                'BART-B': 'bart-base', 'BART-L': 'bart-large', 'mBERT': 'bert-base-multilingual-uncased', 'XLMR-B': 'xlm-roberta-base', 'XLMR-L': 'xlm-roberta-large'}
    model =  json.load(open('./emotion_scores/pretrained_models/'+Category+'_'+names[Model]+'.json',"r"))
    Groups = None
    if Groups == None:
        if sample > 0:
            Groups = np.random.choice(list(model.keys()), sample)
        else:
            Groups = [ word for word in model.keys() ]

  
    word_vectors = np.array([model[w.lower()][0] for w in Groups])
    
    three_dim = PCA(random_state=42).fit_transform(word_vectors)[:,:3]

    # For 2D, change the three_dim variable into something like two_dim like the following:
    data = []
    count = 0
    user_input = Groups
    for i in range (len(user_input)):
               
                trace = go.Scatter3d(
                    x = [word_vectors[i][0]], 
                    y = [word_vectors[i][1]], 
                    z = [word_vectors[i][3]],
                    text = Groups[i] + '('+str(int(word_vectors[i][0]))+','+ str(int(word_vectors[i][1]))+ ','+str(int(word_vectors[i][3]))+')',
                    textposition = "top center",
                    textfont_size = 14,
                    mode = 'markers+text',
                    marker = {
                        'size': 3,
                        'opacity': 0.8,
                        'color': 2
                    }
                )
                
                data.append(trace)
                count +=1


    
# Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        title="BERT-B emotion profiles",
        scene = dict(
        xaxis_title="Negative",
        yaxis_title="Positive",
        zaxis_title="Anger"),
        showlegend=False,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=10,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 12),
        autosize = False,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()


def display_pca_scatterplot_2D(model, user_input=None, words=None, label=None, color_map=None, topn=5, sample=25):
    model =  json.load(open(model,"r"))
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.keys()), sample)
        else:
            words = [ word for word in model.keys() ]
    print(words)

    words =   ["Asians", "Americans", "Jews", "Black people",
      "White people",  
     "White Americans", "Black Americans",  
     "Black men", "White men", "Black women", "White women",  "Asian parents", "Black fathers",  #, "Latinos", "Brazilian women","scots",
     "Asian kids", 
     "Black kids", "White kids"] 
    word_vectors = np.array([model[w.lower()][0] for w in words])
    
    # For 2D, change the three_dim variable into something like two_dim like the following:
    two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0
    user_input = words
    for i in range (len(user_input)):

                trace = go.Scatter(
                    x = [two_dim[i][0]], 
                    y = [two_dim[i][1]],
                    text = words[i],
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 2
                    }
       
                )
                
              
                data.append(trace)
                count +=1
    
# Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
         title="BERT-B emotion profiles",
        scene = dict(
        xaxis_title="Negative",
        yaxis_title="Positive"),
        showlegend=False,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 20),
        autosize = True,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()
    
#display_pca_scatterplot_3D('finetuned1epoch/bert-base/race_bert-base-uncased.json',  sample=0)
import logging
import scipy.stats
import scipy as sp
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

from compute_emotion_scores import compute_emotion_scores
import seaborn as sns

from mpl_toolkits.axes_grid1 import make_axes_locatable

def write_matrix(data, filename, labels=[]):
    with open(filename, "w") as f:
        f.write('# Array shape: {0}\n'.format(data.shape))
        if len(labels) > 0:
            for label in labels:
                f.write(str(label) + "\t")
            f.write("\n")

        i = 0
        for data_slice in data:
            if len(labels) > 0:
                f.write(str(labels[i]) + "\t")
            np.savetxt(f, data_slice, fmt='%-5.3f', newline="\t")

            # Writing out a break to indicate different slices...
            f.write('\n')
            i += 1

### Representational Similarity Analysis
# Calculate similarity matrices for all languages
def get_dists(data, labels=[], ticklabels=[], distance="cosine", save_dir="plots/"):
    #logging.info("Calculating dissimilarity matrix")
    x, C = {}, {}
    # For each list of vectors
    for i in np.arange(len(data)):
        x[i] = data[i]
        # Calculate distances between vectors
        C[i] = 1 - (sp.spatial.distance.cdist(x[i], x[i], distance) + 0.00000000001)
        # Normalize
        C[i] /= C[i].max()
    return x, C


# Compare two or more RDMs
def compute_distance_over_dists(x, C, labels, cat, savedir ):
    logging.info("Calculate correlation over RDMs")
    keys = np.asarray(list(x.keys()))

    spearman = np.zeros((len(keys), len(keys)))
    # pearson = np.zeros((len(keys), len(keys)))
    for i in np.arange(len(keys)):
        for j in np.arange(len(keys)):
            corr_s, corr_p = [], []
            for a, b in zip(C[keys[i]], C[keys[j]]):
                s, _ = sp.stats.spearmanr(a, b)
                p, _ = sp.stats.pearsonr(a, b)
                corr_s.append(s)
                corr_p.append(p)
            spearman[i][j] = np.mean(corr_s)
            # If you prefer Pearson correlation
            # pearson[i][j] = np.mean(corr_p)

    im, cbar = get_plot(spearman, labels, cbarlabel= cat.capitalize() + " - Spearman Correlation")#
    #plt.savefig(savedir+"/RSA_Spearman_"+cat+".png",  bbox_inches='tight',pad_inches = 0 )
    #plt.savefig(savedir+"/RSA_Spearman_all.pdf",  bbox_inches='tight',pad_inches = 0 )
    #plt.show()
    # Save the matrix
    #write_matrix(spearman, savedir + "RSA_Spearman.txt", labels=labels)
    emotion_shift = [np.round(i,2)-1 for i in spearman[0]]
    return emotion_shift, im, cbar

def get_plot(data, labels, title="", cbarlabel="Cosine Distance"):
    plt.rcParams["axes.grid"] = False
    plt.interactive(False)
    fig, ax = plt.subplots(figsize=(7, 7))
    im, cbar= heatmap(data, labels, labels, ax=ax,
                       cmap="Blues", vmin = 0.2, vmax =1, title=title, cbarlabel=cbarlabel)
    fig.tight_layout()
    return fig, cbar

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", title="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Only plot upper half
    mask = np.tri(data.shape[0], k=-1)
    data = np.ma.array(data, mask=mask)
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    ax.set_title(title, pad=50.0, fontsize=13)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax, orientation='horizontal', **cbar_kw)
    cax.set_xlabel(cbarlabel, fontsize=15)
    cax.tick_params(axis='y', labelsize=15)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels, fontsize=15)
    ax.set_yticklabels(row_labels, fontsize=15)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    ax.set_xticks(np.arange(0, data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(0, data.shape[0] + 1) - 0.5, minor=True)

    return im, cbar


params = 'all_eng'
if params == 'all_eng':
    models =['bert-base-uncased', 'bert-large-uncased', 'roberta-base', 'roberta-large', 'bart-base', 'bart-large', 'bert-base-multilingual-uncased', 'xlm-roberta-base', 'xlm-roberta-large']
    labels = ['BERT-B', 'BERT-L', 'RoBERTa-B', 'RoBERTa-L', 'BART-B', 'BART-L', 'mBERT', 'XLMR-B', 'XLMR-L']
    output_dir = 'all_eng_freq'     
elif params == 'finetuned':
    models =['bert-base-uncased',  'new_yorker', 'guardian', 'reuters', 'fox', 'breitbart']
    labels = ['BERT-B', 'NewYorker', 'Guardian', 'Reuters', 'FoxNews', 'Breitbart']
    output_dir = 'finetuned1epoch/bert-base'     
elif params == 'ablation':
    models =['bert-base-uncased', 'new_yorker', 'guardian', 'reuters', 'fox', 'breitbart']
    labels = ['BERT-B', 'NewYorker', 'Guardian', 'Reuters', 'FoxNews', 'Breitbart']
    output_dir = 'finetuned-half'     

def comparison_across_models(cat_of_interest):

    models =['bert-base-uncased', 'bert-large-uncased', 'roberta-base', 'roberta-large', 'bart-base', 'bart-large', 'bert-base-multilingual-uncased', 'xlm-roberta-base', 'xlm-roberta-large']
    labels = ['BERT-B', 'BERT-L', 'RoBERTa-B', 'RoBERTa-L', 'BART-B', 'BART-L', 'mBERT', 'XLMR-B', 'XLMR-L']
    output_dir = 'emotion_scores/pretrained_models/'    
    shifts, vectors = [], []
    for i in range(0, len(models)):  
        array, targets = compute_emotion_scores(models[i],  cat_of_interest,  output_dir, labels[i])
        vectors.append(array)

    x, C = get_dists(vectors, labels=labels, ticklabels=targets, distance="cosine")
    # Calculate RSA over all languages
    emotion_shift, im, cbar = compute_distance_over_dists(x, C, labels, cat_of_interest, output_dir)
    return  im, cbar

def comparison_within_models(cat_of_interest, model, finetuned='finetuned1epoch'):
    
    names = {'BERT-B': 'bert-base-uncased', 'RoBERTa-B': 'roberta-base',
                'BART-B': 'bart-base', 'mBERT': 'mbert', 'XLMR-B': 'xlm-roberta-base'}
    models = [names[model]] + ['new_yorker', 'guardian', 'reuters', 'fox', 'breitbart']
    labels = [model] + ['NewYorker', 'Guardian', 'Reuters', 'FoxNews', 'Breitbart']

    shifts, vectors = [], []
    for i in range(0, len(models)):  
        
        if finetuned !='finetuned1epoch' and models[i] != 'bert-base-uncased':
            output_dir = 'emotion_scores/' + finetuned 
        else: 
            output_dir = 'emotion_scores/finetuned1epoch/'+ names[model]     
        array, targets = compute_emotion_scores(models[i],  cat_of_interest,  output_dir, labels[i])
        vectors.append(array)    

    x, C = get_dists(vectors, labels=labels, ticklabels=targets, distance="cosine")
    # Calculate RSA over all languages
    emotion_shift, im, cbar = compute_distance_over_dists(x, C, labels, cat_of_interest, output_dir)
    return  im, cbar

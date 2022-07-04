import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import BoundaryNorm, ListedColormap
from netneurotools import stats as nnstats
import seaborn as sns
import itertools
# from IPython import embed as shell # Olympia's debugging

#########################

def voxel_means (df):
    #Average over all voxels per ROI
    df = df.iloc[:, 1:]
    df = df.drop('mask_idx', axis=1)
    df_reduced = df.groupby(['session', 'subject', 'letter', 'trial_type_color', 'brain_labels'], as_index=False).mean()
    df_reduced.to_csv('df_reduced.csv')
    return df_reduced

def merge_files(df1,df2):
    df_merged = df1.merge(df2, how='inner', on=['subject', 'letter', 'trial_type_color'])
    df_merged.to_csv('df_merged.csv')
    return df_merged

def to_matrix(df, c):
    #convert to correlation matrix
    df = df.groupby('brain_labels').mean()
    df_values = df[c]
    df_values_t = df_values.transpose()
    return df_values_t.corr('pearson')

def stack(df):
    #stack values to use in testing.
    df = df.stack().rename_axis(('Region x', 'Region y')).reset_index(name='value').drop_duplicates(subset='value', keep='first') #.sort_values('value', ascending=False)
    df = df.loc[~(df['Region x'] == df['Region y'])]
    df.reset_index(drop=True, inplace=True)
    return df

def split_data(df, col_name, split):
    #splits the data based on split
    df_split1 = df[df[col_name] == split[0]]
    df_split2 = df[df[col_name] == split[1]]
    return df_split1, df_split2

def compare(df1, df2):
    # subtract df1 from df2
    return df2.sub(df1)

def fischer_transform(r):
    return 0.5*np.log((1+r)/(1-r))

def tril_to_square_matrix(n_rois,a):
    # Covert from lower diagonal to square matrix form
    # returns a square matrix with size = len(a),len(a)
    # https://stackoverflow.com/questions/16444930/copy-upper-triangle-to-lower-triangle-in-a-python-matrix
    M = np.ones((n_rois, n_rois))          # create square matrix all 1s
    M[np.triu_indices(n_rois, 1)] = a       # fill in upper diagonal
    # fill lower based on upper
    i_lower = np.tril_indices(n_rois, -1)
    M[i_lower] = M.T[i_lower]
    return M

def rq():
    # testing research questions
    # compares the stacked dataframes for these conditions, test the difference against 0 using a 1 sample perm test

    # load the list of roi's for plotting
    label_list = pd.read_csv("harvard_oxford_cortical_labels_new.csv").sort_values('new_order')

    # load each session's data frame
    d1 = pd.read_csv("df_post_color_untrained_stacked.csv")
    d2 = pd.read_csv("df_post_color_trained_stacked.csv")
    
    try: # drop all unnamed columns
        d1 = d1.loc[:, ~d1.columns.str.contains('^Unnamed')]
        d2 = d2.loc[:, ~d2.columns.str.contains('^Unnamed')]
        d1 = d1.loc[:, ~d1.columns.str.contains('subject')]
        d2 = d2.loc[:, ~d2.columns.str.contains('subject')]
    except:
        pass
    
    # take the difference (session2-session1)
    compared = compare(d1,d2)
    
    # transform the difference to Z scores (still with all subjects, you only take the mean for plotting)
    compared = fischer_transform(compared)

    # covert NaN to zero
    compared = pd.DataFrame(np.nan_to_num(compared))

    # perm test (note this has to be on the data frame with individual participants data, NOT the means)
    stat,pval = nnstats.permtest_1samp(compared, 0.0)

    # convert to square matrix for plotting
    mean_compared = compared.mean()
    corr_matrix = tril_to_square_matrix(label_list.shape[0],np.array(mean_compared))
    pval_matrix = tril_to_square_matrix(label_list.shape[0],np.array(pval))
    pd.DataFrame(corr_matrix).to_csv('corr_matrix_post_color.csv')
    pd.DataFrame(pval_matrix).to_csv('pval_syn_post_color.csv')

    # plotting

    labels = label_list['ROI']
    corr_colors = ['white', 'dimgrey']
    corr_cmap = ListedColormap(corr_colors)
    corr_bounds = [-1, 1]
    corr_norm = BoundaryNorm(corr_bounds, ncolors=len(corr_colors))

    pval_norm = matplotlib.colors.Normalize(0, 0.05)
    pval_colors = [[pval_norm(0), "darkorange"],
              [pval_norm(0.025), "violet"],
              [pval_norm(0.05), "lightsteelblue"]]
    pval_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", pval_colors)

    #creating figure and color bar
    fig, ax = plt.subplots(figsize=(20,13))
    cb_axes = fig.add_axes([0.03, 0.882, 0.15, 0.02])

    #creates text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.figtext(0.0024, 0.064, 'Frontal Lobe', color='saddlebrown', fontsize=12)
    plt.figtext(0.0, 0.0, '\n Association Areas \n \n \n', bbox=props, fontsize=12)
    plt.figtext(0.0024, 0.032, 'Temporal Lobe', color = 'darkcyan', fontsize=12)
    plt.figtext(0.0024, 0.016, 'Occipital Lobe', color = 'darkmagenta', fontsize=12)
    plt.figtext(0.0024, 0.00, 'Parietal Lobe', color = 'maroon' , fontsize=12)

    #plots correlation matrix
    ax.set_title("Change in correlation coefficient between VWFA, V4 and Parietal Lobe for \n trained color letters pre-training vs trained color letters post-training", fontsize=25)

    pval_boolean = pval_matrix >= 0.05  # for mask: If passed, data will not be shown in cells where mask is True. Cells with missing values are automatically masked.
    neg = corr_matrix > 0.00 #for mask for negative correlation values
    pos = corr_matrix <= 0.00 #for mask for positive correlation values
    sns.heatmap(corr_matrix, ax=ax, mask=pos, annot=True, annot_kws={"size": 7, "weight": 550}, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap=corr_cmap, norm=corr_norm, linecolor='dimgrey', linewidths=0.003, cbar=False)
    sns.heatmap(corr_matrix, ax=ax, mask=neg, annot=True, annot_kws={"size": 7, "style": 'italic'}, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap=corr_cmap, norm=corr_norm, linecolor='dimgrey', linewidths=0.003, cbar=False)
    sns.heatmap(pval_matrix, ax=ax, mask=pval_boolean, cmap=pval_cmap, xticklabels=labels, yticklabels=labels, vmax=0.05, cbar=True, norm=pval_norm, cbar_ax=cb_axes, cbar_kws={ "orientation": "horizontal", "label": "P-values", "shrink":0.3, "ticklocation":"top"})
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha='right')

    #color labels
    for idx, tick_label in enumerate(ax.get_xticklabels()):
        if idx < 9:
            tick_label.set_color('saddlebrown')
        if 9 <= idx < 17 or 39 <= idx < 42:
            tick_label.set_color('black')
        if 17 <= idx < 33:
            tick_label.set_color('darkcyan')
        if 33 <= idx < 42:
            tick_label.set_color('darkmagenta')
        if idx >= 42:
            tick_label.set_color('maroon')

    for idx, tick_label in enumerate(ax.get_yticklabels()):
        if idx < 9:
            tick_label.set_color('saddlebrown')
        if 9 <= idx < 17 or 39 <= idx < 42:
            tick_label.set_color('black')
        if 17 <= idx < 33:
            tick_label.set_color('darkcyan')
        if 33 <= idx < 42:
            tick_label.set_color('darkmagenta')
        if idx >= 42:
            tick_label.set_color('maroon')

    fig.savefig('syn_post_untrained_trained_color.jpg', bbox_inches='tight')
    print('success')


def data_processing():
    # run all the necessary processing on the data frames. 
    # import, reduce, merge, split by condition.
    # Note that once the FINAL output has been generated, there is no need to continue to run this function!
    
    letters_conditions = pd.read_csv("task-rsa_letters_conditions.tsv",         sep='\t')
    ev_conditions      = pd.read_csv("task-rsa_letters_ev_conditions.tsv",      sep='\t', index_col=False, usecols=["ev","letter","trial_type_color"])
    df_roi             = pd.read_csv("task-rsa_letters_timeseries_rois.tsv",    sep='\t')

    df_roi = df_roi[df_roi.trial_type_color != 'grey']

    c = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    df_reduced = voxel_means(df_roi)
    df_merged = merge_files(df_reduced, letters_conditions)

    participants_list = np.unique(df_roi.subject)

    df_pre, df_post = split_data(df_merged, 'session', [1,2])
    df_pre_black, df_pre_color = split_data(df_pre, 'trial_type_color', ['black', 'color'])
    df_post_black, df_post_color = split_data(df_post, 'trial_type_color', ['black', 'color'])

    df_pre_black_trained, df_pre_black_untrained = split_data(df_pre_black, 'trial_type_letter', ['trained', 'untrained'])
    df_pre_color_trained, df_pre_color_untrained = split_data(df_pre_color, 'trial_type_letter', ['trained', 'untrained'])

    df_post_black_trained, df_post_black_untrained = split_data(df_post_black, 'trial_type_letter', ['trained', 'untrained'])
    df_post_color_trained, df_post_color_untrained = split_data(df_post_color, 'trial_type_letter', ['trained', 'untrained'])

    conditions = [df_pre_black_untrained, df_pre_black_trained, df_pre_color_untrained, df_pre_color_trained,
                  df_post_black_untrained, df_post_black_trained, df_post_color_untrained, df_post_color_trained]

    conditions_names = ['df_pre_black_untrained', 'df_pre_black_trained', 'df_pre_color_untrained', 'df_pre_color_trained',
                        'df_post_black_untrained', 'df_post_black_trained', 'df_post_color_untrained', 'df_post_color_trained']

    for dfIdx, df in enumerate(conditions):
        df_output = pd.DataFrame(columns=list(itertools.combinations(np.unique(df.brain_labels), 2)))

        for p in participants_list:
            df_p = df[df['subject'] == p]
            df_matrix = to_matrix(df_p, c)
            df_stacked = stack(df_matrix)
            df_output.loc['{}'.format(p)] = df_stacked.value.to_numpy()

        # add 'subject' as column name
        df_output.reset_index(inplace=True)
        df_output.rename(columns={'index':'subject'}, inplace=True)
        df_output.to_csv('{}_stacked.csv'.format(conditions_names[dfIdx]))
        # NOTE that once these stacked condition files are created, I don't need to run the above code anymore. 
    print('success: data_processing!')

###########################

#data_processing()
rq()


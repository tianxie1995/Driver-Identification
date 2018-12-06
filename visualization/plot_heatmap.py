import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import initiate.config as config


def plot_correction_heatmap(RD_dict, correction_matrix, mean_accuracy):
    '''
    gnerate correction heatmap for 1000 times experiment 
    :param RD_dict
    :param correction_matrix (8x1000)
    :param mean_accuracy
    '''
    print('correction matrix shape', correction_matrix.shape)
    unnomalized_heatmap_matrix = np.zeros((8, 8))
    for col in correction_matrix.T:
        local_corection_matrix = np.zeros_like(unnomalized_heatmap_matrix)
        for i, predicted_driver in enumerate(col):
            local_corection_matrix[int(predicted_driver), i] = 1
        assert local_corection_matrix.shape == (
            8, 8), 'Local Correction Matrix Shape is wrong'
        unnomalized_heatmap_matrix += local_corection_matrix
    normalized_heatmap_matrix = unnomalized_heatmap_matrix / \
        correction_matrix.shape[1]
    plt.figure(figsize=(10, 10))
    sns.heatmap(normalized_heatmap_matrix,
                annot=True, cmap="plasma",
                xticklabels=list(RD_dict.keys()), yticklabels=list(RD_dict.keys()),
                square=True, cbar_kws={'label': 'Percentage', 'shrink': 0.8})
    xlocs, labels = plt.xticks()
    ylocs, labels = plt.yticks()
    plt.yticks(xlocs, labels=list(
        RD_dict.keys()), fontsize=8)
    plt.xticks(ylocs, labels=list(
        RD_dict.keys()), fontsize=8)
    plt.xlabel('Driver ID')
    plt.ylabel('Driver ID')
    plt.title('Mean Accuracy among {:,} experiments ={:.2%}'.format(
        correction_matrix.shape[1], mean_accuracy), fontweight='bold')
    save_figure_path = os.path.join(
        config.figures_folder, config.heatmap_folder)
    if not os.path.exists(save_figure_path):
        os.mkdir(save_figure_path)
    plt.savefig(os.path.join(save_figure_path, "heatmap" + ".png"))
    plt.close()

import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap


## Plotting for zero_out (InpxGrad)
def plot_cifar_grad(grid,
                     folderName,
                     row_labels_left,
                     row_labels_right,
                     col_labels,
                     file_name=None,
                     dpi=32,
                     save=True,
                     rescale=True, flag=0
                     ):

    plt.rcParams['axes.linewidth'] = 0.0  # set the value globally
    plt.rcParams.update({'font.size': 5})
    plt.rc("font", family="sans-serif")
    plt.rc("axes.spines", top=True, right=True, left=True, bottom=True)
    image_size = (grid[0][0]).shape[0]
    nRows = len(grid)
    nCols = len(grid[0])
    tRows = nRows + 2  # total rows
    tCols = nCols + 1  # total cols
    wFig = tCols
    hFig = tRows  # Figure height (one more than nRows becasue I want to add xlabels to the top of figure)
    fig, axes = plt.subplots(nrows=tRows, ncols=tCols, figsize=(wFig, hFig))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    axes = np.reshape(axes, (tRows, tCols))
    #########

    # Creating colormap
    uP = cm.get_cmap('Reds', 129)
    dowN = cm.get_cmap('Blues_r', 128)
    newcolors = np.vstack((
        dowN(np.linspace(0, 1, 128)),
        uP(np.linspace(0, 1, 129))
    ))
    cMap = ListedColormap(newcolors, name='RedsBlues')
    cMap.colors[257//2, :] = [1, 1, 1, 1]

    #######
    scale = 0.99
    fontsize = 15
    o_img = grid[0][0]
    for r in range(tRows):
        # if r <= 1:
        for c in range(tCols):
            ax = axes[r][c]
            l, b, w, h = ax.get_position().bounds
            ax.set_position([l, b, w * scale, h * scale])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            if r > 0 and c > 0 and r < tRows - 1:
                img_data = grid[r - 1][c - 1]
                abs_min = np.amin(img_data)
                abs_max = np.amax(img_data)
                abs_mx = max(np.abs(abs_min), np.abs(abs_max))
                r_abs_min = round(np.amin(img_data), 2)
                r_abs_max = round(np.amax(img_data), 2)
                r_abs_mx = round(max(np.abs(abs_min), np.abs(abs_max)), 2)

                # Orig Image
                if r == 1 and c == 1:
                    im = ax.imshow(img_data, interpolation='none')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                else:
                    # im = ax.imshow(o_img, interpolation='none', cmap=cMap, vmin=-1, vmax=1)
                    im = ax.imshow(img_data, interpolation='none', cmap=cMap)  # , vmin=-1, vmax=1)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    # save 1

                zero = 0
                if r < tRows:
                    if col_labels != []:
                        # import ipdb
                        # ipdb.set_trace()
                        if c == 1:
                            ax.set_xlabel(col_labels[c - 1],
                                          horizontalalignment='center',
                                          verticalalignment='bottom',
                                          fontsize=9, labelpad=17)
                        else:
                            temp_label = col_labels[c - 1].split(' ')
                            ax.set_xlabel(' '.join(temp_label[:2]) + '\n' + ' '.join(temp_label[-2:]),
                                          horizontalalignment='center',
                                          verticalalignment='bottom',
                                          fontsize=9, labelpad=21)
                if c == tCols - 2:
                    if row_labels_right != []:
                        txt_right = [l + '\n' for l in row_labels_right[r - 1]]
                        ax2 = ax.twinx()
                        ax2.set_xticks([])
                        ax2.set_yticks([])
                        ax2.spines['top'].set_visible(False)
                        ax2.spines['right'].set_visible(False)
                        ax2.spines['bottom'].set_visible(False)
                        ax2.spines['left'].set_visible(False)
                        ax2.set_ylabel(''.join(txt_right), rotation=0,
                                       verticalalignment='center',
                                       horizontalalignment='left',
                                       fontsize=fontsize)
                if c == 1:  # (not c - 1) or (not c - 2) or (not c - 4) or (not c - 6):
                    if row_labels_left != []:
                        txt_left = [l + '\n' for l in row_labels_left[r - 1]]
                        ax.set_ylabel(''.join(row_labels_left[0]),
                                      # rotation=0,
                                      # verticalalignment='center',
                                      # horizontalalignment='center',
                                      fontsize=fontsize)
                # else:
                if c == tCols-1 and flag==0:  # > 1 # != 1:
                    w_cbar = 0.009
                    h_cbar = h * 0.9  # scale
                    b_cbar = b
                    l_cbar = l + scale * w + 0.001
                    cbaxes = fig.add_axes([l_cbar + 0.015, b_cbar + 0.015, w_cbar, h_cbar])
                    cbar = fig.colorbar(im, cax=cbaxes)
                    cbar.outline.set_visible(False)
                    cbar.ax.tick_params(labelsize=5, width=0.2, length=1.2, direction='inout', pad=0.5)
                    tt = 1
                    # cbar.set_ticks([])
                    cbar.set_ticks([-tt, zero, tt])
                    # cbar.set_ticklabels([-1, zero, 1])
    if save:
        dir_path = folderName
        print(f'Saving figure to {os.path.join(dir_path, file_name)}')

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(os.path.join(dir_path, file_name), dpi=dpi / scale, transparent=True, bbox_inches='tight', pad_inches=0)
        # plt.savefig(os.path.join(dir_path, file_name), orientation='landscape', dpi=dpi / scale, transparent=False,
        #             frameon=False)
        plt.close(fig)
    else:
        plt.show()

    plt.close(fig)

#########################
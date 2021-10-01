import pandas as pd
import numpy as np
import os, copy, sys
import shutil
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageOps
# from skimage import feature
from plotHelper import plot_surgery_by_layer, plot_concept_importance, layerNames, scatterPlot, scatterPlot2, scatterPlotPair
import torch
import argparse
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg
import random
import ipdb

def check_path(path):
    sep_path = path.split('/')
    test_path = sep_path[0]
    # if input is a dir+file name, delete file name
    if os.path.isfile(path):
        del sep_path[-1]
    for seg in sep_path[1:]:
        test_path = test_path+f'/{seg}'
        if not os.path.isdir(test_path):
            os.mkdir(test_path)

# montage gradient plots
def montage_tv_train_grad(varies, model='ResNet18', img_path='result/grad_img',
                          montage_path='result/grad_img/montage'
                          ):
    files = os.listdir(f'result/grad_img/{model}_s')
    files.sort()
    for file in files:
        os.system(
            f"montage -pointsize 30 -label 'Original' result/grad_img/org/{file} "
            f"-label 'Standard' {img_path}/{model}_s/{file} "
            f"-label {varies[0]} {img_path}/{model}_{varies[0]}/{file} "
            f"-label {varies[1]} {img_path}/{model}_{varies[1]}/{file} "
            f"-label {varies[2]} {img_path}/{model}_{varies[2]}/{file} "
            f"-label {varies[3]} {img_path}/{model}_{varies[3]}/{file} "
            f"-label 'Robust' {img_path}/{model}_robust_eps_1/{file} "
            f"-tile 7x1 -geometry +0+0 {montage_path}/{file}")

# plot gradient with chirag's colormap
def plot_grad(file_path, out_path):
    check_path(out_path)
    files = os.listdir(file_path)
    if file_path.split('/')[-1] == 'org':
        files.sort()
        for file in files[:10]:
            img = mpimg.imread(f'{file_path}/{file}')
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'{out_path}/{file}')

    else:
        for idx, img in enumerate(files):
            if idx > 10:
                break
            grad_img = np.load(f'{file_path}/{img}')
            grad_img = np.mean(grad_img, axis=-1)
            grad_img = grad_img / np.max(np.abs(grad_img))
            # normalize
            # for channel in range(3):
            #     if grad_img[:, :, channel].min() < 0:
            #         grad_img[:, :, channel] += abs(grad_img[:, :, channel].min())
            #     grad_img[:, :, channel] = grad_img[:, :, channel] / grad_img[:, :, channel].max()
            # Creating colormap
            uP = cm.get_cmap('Reds', 129)
            dowN = cm.get_cmap('Blues_r', 128)
            newcolors = np.vstack((
                dowN(np.linspace(0, 1, 128)),
                uP(np.linspace(0, 1, 129))
            ))
            cMap = ListedColormap(newcolors, name='RedsBlues')
            cMap.colors[257 // 2, :] = [1, 1, 1, 1]

            plt.imshow(grad_img, interpolation='none', cmap=cMap)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f'{out_path}/{img[:-4]}.jpg')



# find super intersect between different dataset
def find_super_inter_dataset():
    clean_csv = pd.read_csv('dataset/correct_csv/alexnet_clean_inter.csv', index_col=0)
    noisy_csv = pd.read_csv('dataset/correct_csv/alexnet_gaussian_inter.csv', index_col=0)

    clean_list = []
    noisy_list = []
    for _, row in clean_csv.iterrows():
        clean_list.append(row.File.split('/')[-1])
    for _, row in noisy_csv.iterrows():
        noisy_list.append(row.File.split('/')[-1])

    clean_frame = pd.DataFrame(clean_list, columns=['File'])
    noisy_frame = pd.DataFrame(noisy_list, columns=['File'])
    inter_frame = pd.merge(clean_frame, noisy_frame, how='inner')
    inter_list = inter_frame.File.to_list()

    true_table = []
    for _, row in clean_csv.iterrows():
        if row.File.split('/')[-1] not in inter_list:
            true_table.append(False)
        else:
            true_table.append(True)
    clean_inter = clean_csv[true_table]
    clean_inter.to_csv('dataset/correct_csv/alexnet_clean_noise_inter_cleanpath.csv')

    true_table = []
    for _, row in noisy_csv.iterrows():
        if row.File.split('/')[-1] not in inter_list:
            true_table.append(False)
        else:
            true_table.append(True)
    noisy_inter = noisy_csv[true_table]
    noisy_inter.to_csv('dataset/correct_csv/alexnet_clean_noise_inter_noisepath.csv')


# simple version of finding intersection
def find_intersect(path1, path2):
    data1 = pd.read_csv(path1, index_col=0).File
    data2 = pd.read_csv(path2, index_col=0).File
    intersect = pd.merge(data1, data2, how='inner')
    return intersect

# string to bool value
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# find stats for each network
def dissect_label_stats(path, network):
    files = os.listdir(path)
    indexName = [a.replace('.csv', '') for a in files]
    stats = pd.DataFrame(np.zeros((len(indexName), 1)), index=indexName, columns=['Init'])
    for each_file in files:
        index = each_file[:-4]
        file_dir = os.path.join(path, each_file)
        rawData = pd.read_csv(file_dir)
        for idx, row in rawData.iterrows():
            if stats.columns.isin([row.label]).any():
                stats.loc[index, row.label] += 1
            else:
                stats[row.label] = 0
                stats.loc[index, row.label] += 1
    stats.drop(columns=['Init'], inplace=True)
    stats.loc['Sum'] = stats.sum()
    stats.sort_values('Sum', axis=1, ascending=False, inplace=True)
    stats.to_csv("result/tally/"+network+"_summary.csv", index=True)


''' Copy images in a list to target folder'''
def copy_img(img_list, out_dir):
    check_path(out_dir)
    for each_file in img_list:
        shutil.copy(each_file, out_dir)


''' Find interection of given two pandas dataframe
Input: pandas DataFrame *2
Output: pandas DataFrame
'''
def find_intersect_diff_stats(path1, path2, save_name=None):
    # read and clean the data
    data1 = pd.read_csv(path1, index_col=0)
    data2 = pd.read_csv(path2, index_col=0)
    obs_data = pd.DataFrame(columns=['File', 'Acc', 'Class'])

    if len(data1) > len(data2):
        data_org = data1
        data_after = data2
    else:
        data_org = data2
        data_after = data1

    # find intersect
    pbar = tqdm(total=len(data_after))
    pbar.set_description('Finding intersection')
    for idx, row in data_after.iterrows():
        if data_org.File.isin([row.File]).any():
            obs_data = obs_data.append(pd.Series([row.File, data_org.loc[idx, 'Confidence'] - row.Confidence, row.Class],
                                      index= ['File', 'Acc', 'Class']), ignore_index=True)
        pbar.update(1)
    pbar.close()
    obs_data.sort_values(by='Acc', inplace=True, ascending=False)
    obs_data.reset_index(drop=True, inplace=True)
    # save a bb for faster read
    if save_name is not None:
        obs_data.to_csv(save_name, index=True)
    return obs_data



''' Function that plot images from image list and confidence list
Input : [ImageList, DataList] or (Pandas DataFrame), output path
Output: Single image
'''
def display_from_stats(dis_data, save_path, topk=100, tile=(10, 10), img_size=224):
    # if input is a list
    if isinstance(dis_data, list):
        if len(dis_data[0]) < topk:
            topk = len(dis_data)
        imageList = dis_data[0]
        confidence = dis_data[1]
    else:
        if len(dis_data) < topk:
            topk = len(dis_data)
        dis_data = dis_data[:topk]
        imageList = dis_data.File.to_list()
        confidence = dis_data.Confidence.to_list()
    font_size = 30
    new_img = Image.new('RGB', (img_size*tile[0], (img_size+font_size)*tile[1]))
    for i, img_file in enumerate(imageList):
        image = Image.open(img_file).convert('RGB')
        image = image.resize((img_size, img_size))
        # generate empty space for confidence socre
        add_conf_space = Image.new('RGB', (img_size, img_size + font_size), (255, 255, 255))
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", font_size, encoding="unic")
        add_conf_space.paste(image, (0, 0))
        # put confidence score input the image
        draw = ImageDraw.Draw(add_conf_space)
        (x_msg, y_msg) = (30, 224)
        if isinstance(confidence[i], str):
            message = confidence[i]
        else:
            message = str(round(float(confidence[i]), 4))
        draw.text((x_msg, y_msg), message, fill=0, font=font)
        # put new image to proper location
        y = (i//tile[0])*(img_size+font_size)
        x = (i % tile[0])*img_size
        new_img.paste(add_conf_space, (x, y))
    new_img.save(save_path)
    return save_path

''' Read labels in a layer'''
def layer_labels(file):
    data = pd.read_csv(file, index_col=0)
    data.sort_values(by='unit', inplace=True)
    return data.label.to_list()


''' Calculate the mean of given npy file and return pandas DataFrame'''
def find_mean(file):
    data = np.load(file)
    mean = np.mean(data, axis=0)
    layer_len, _, _ = mean.shape
    difference = []
    for unit in range(layer_len):
        difference.append(mean[unit].sum())
    diff_frame = pd.DataFrame(difference, columns=['Score'])
    return diff_frame


''' Find statistical result after surgery'''
def find_stats_after_surgery(model_path, label_path, save_path):
    stat_files = os.listdir(model_path)
    for npy_file in stat_files:
        layer_name = npy_file.split(".")[0]
        score_frame = find_mean(f'{model_path}/{npy_file}')
        test = layer_labels(f'{label_path}/tally{layer_name}.csv')
        score_frame['Label'] = test
        score_frame['Unit'] = range(len(score_frame))
        score_frame.sort_values(by="Score", ascending=False, inplace=True, ignore_index=True)
        score_frame.to_csv(f'{save_path}/{layer_name}.csv')


def rename_by_idx(dir, spilt='_'):
    os.chdir(dir)
    files = os.listdir()
    for each_file in files:
        name = each_file.split(spilt)
        name[0] = name[0].zfill(3)
        new_name = f'{name[0]}_{name[1]}'
        os.rename(each_file, new_name)


''' Find the difference confidence '''
def find_confidence_difference(target_data, org_data, relative=False, full_path=False):
    conf_diff = []
    # keep name only
    if full_path:
        img_name_list = []
        for idx, row in org_data.iterrows():
            img_name_list.append(row.File.split('/')[-1])
        org_data.File = img_name_list

        # img_name_list = []
        # for idx, row in target_data.iterrows():
        #     img_name_list.append(row.File.split('/')[-1])
        # target_data.File = img_name_list

    for _, row in target_data.iterrows():
        if relative:
            org = org_data[org_data.File == row.File].Confidence.values[0]
            after = row.Confidence
            changes = (org - after)/org
            conf_diff.append(changes)
        else:
            if full_path:
                img_name = row.File.split('/')[-1]
                conf_diff.append((org_data[org_data.File == img_name].Confidence - row.Confidence).values[0])
            else:
                conf_diff.append((org_data[org_data.File == row.File].Confidence - row.Confidence).values[0])

    return conf_diff


''' Find accuracy '''
def find_accuracy(target_folder, org_result, by='channelID', keep=False, orgSet=False):
    file_list = os.listdir(target_folder)
    org_length = len(pd.read_csv(org_result))
    if not keep:
        accuracy = pd.DataFrame(np.zeros((len(file_list), 2)), columns=['Target', 'Accuracy_Drop'])
    else:
        accuracy = pd.DataFrame(np.zeros((len(file_list), 2)), columns=['Target', 'Accuracy'])
    common_channels = []
    for idx, file in enumerate(file_list):
        channel_id = file.split('_')[0]
        if by == 'concept':
            concept = file.split('_')
            # concept.pop(0)
            concept.pop(-1)
            concept = "_".join(concept)
        item_length = len(pd.read_csv(f'{target_folder}/{file}'))
        # in original dataset
        if orgSet:
            acc_drop = org_length/50000 - (1-item_length/50000)
        # in intersect dataset
        else:
            acc_drop = item_length/org_length
        if by == 'channelID':
            accuracy.iloc[idx] = channel_id, acc_drop
        elif by == 'concept':
            accuracy.iloc[idx] = concept, acc_drop
            common_channels.append(int(file.split('_')[-1][:-4]))

    if by == 'concept':
        accuracy['Counts'] = common_channels
    return accuracy


def find_accuracy_stats(networks, operation, index_by, layer='11', keep=False):
    accuracy_stats = pd.DataFrame(columns=networks)
    for network in networks:
        net_stats = pd.read_csv(f'result/tally/{network}/tally{layer}.csv', index_col=False)
        if not keep:
            folder = f'result/wrong_csv_alblation/zero_out/{network}/{operation}'
            org_path = 'dataset/correct_csv/alexnet_inter.csv'
            accuracy_drop = find_accuracy(folder, org_path, index_by)
            for idx, row in accuracy_drop.iterrows():
                if len(row) == 2:
                    accuracy_stats.loc[row.Target, network] = row.Accuracy_Drop
                elif len(row) == 3:
                    accuracy_stats.loc[row.Target, network] = row.Accuracy_Drop
                    accuracy_stats.loc[row.Target, 'Count'] = row.Counts
        else:
            folder = f'result/correct_csv_alblation/keep/{network}/{operation}'
            org_path = 'dataset/correct_csv/alexnet_inter.csv'
            accuracy = find_accuracy(folder, org_path, index_by, keep)
            for idx, row in accuracy.iterrows():
                if len(row) == 2:
                    accuracy_stats.loc[row.Target, network] = row.Accuracy
                elif len(row) == 3:
                    accuracy_stats.loc[row.Target, network] = row.Accuracy
                    accuracy_stats.loc[row.Target, 'Count'] = row.Counts
        if index_by == 'channelID':
            net_stats.sort_values(by='unit', inplace=True)
            net_stats.reset_index(drop=True, inplace=True)
            accuracy_stats[f'Concept_{network}'] = net_stats.label
        elif index_by == 'concept':
            accuracy_stats[f'Concept_{network}'] = accuracy_stats.index
    return accuracy_stats


# fix naming issue in alblation experiment
def fix_name(model, layer, patient_path, file_extension='csv'):
    layer_stats = pd.read_csv(f'result/tally/{model}/tally{layer}.csv')
    file_list = copy.deepcopy(os.listdir(patient_path))
    os.chdir(patient_path)
    for file in file_list:
        channel_id = file.split('_')[0]
        correct_label = layer_stats[layer_stats.unit == (int(channel_id) + 1)].label.values[0]
        name = f'{channel_id}_{correct_label}.{file_extension}'
        os.rename(file, name)


# plot output images after alblation
    '''Display images after alblation
    Input: file_path: csv file path 
            networks: network names
            specification: folder name of specific operation
            save_path: path that save results
    '''
def show_images_after_alblation(alblation_type, networks, specification, reletive=False, abs=False, full_path=False, data_type=None):
    if alblation_type == 'zero_out':
        file_path = f'result/wrong_csv_alblation/{alblation_type}'
    elif alblation_type == 'keep':
        file_path = f'result/correct_csv_alblation/{alblation_type}'
    for network in networks:
        path = f'{file_path}/{network}/{specification}'
        files = os.listdir(path)
        if data_type is None:
            comp_data = pd.read_csv(f'dataset/correct_csv/{network}.csv')
        else:
            comp_data = pd.read_csv(f'dataset/correct_csv/{network}_{data_type}.csv')
        pbar = tqdm(total=len(files))
        pbar.set_description(f"Ploting images for {network}")
        for csv_file in files:
            data_path = f'{path}/{csv_file}'
            save_path = f'result/alblation_plots/{alblation_type}/{network}/{specification}/{csv_file[:-4]}.jpg'
            check_path(save_path)
            img_data = pd.read_csv(data_path, index_col=0)
            # img_data.sort_values(by='Confidence', ascending=False, inplace=True)
            file_list = img_data.File
            if alblation_type == 'zero_out':
                conf_list = find_confidence_difference(img_data, comp_data, reletive, full_path=full_path)
            elif alblation_type == 'keep':
                conf_list = img_data.Confidence
            if abs:
                conf_list = abs(conf_list)
            # new image data with confidence difference
            img_data = pd.DataFrame(file_list, columns=['File'])
            img_data['Confidence'] = conf_list
            img_data.sort_values(by='Confidence', ascending=False, inplace=True)

            conf_list = img_data.Confidence.values
            conf_list_in_percentage = []
            for i in range(len(conf_list)):
                drop = round(conf_list[i]*100, 2)
                conf_list_in_percentage.append(f'{drop}%')
            img_data['Confidence'] = conf_list_in_percentage
            display_from_stats(img_data, save_path, 100, (10, 10))
            pbar.update(1)
        pbar.close()

# remove not intersect data
def cleanData():
    layer_data_1 = pd.read_csv(f'result/tally/alexnet/tally11.csv')
    layer_data_2 = pd.read_csv(f'result/tally/alexnet-r/tally11.csv')
    labels1 = layer_data_1.label
    labels2 = layer_data_2.label
    labels = pd.merge(labels1, labels2, how='inner')
    labels = labels.label.unique()

    path1 = 'result/wrong_csv_alblation/zero_out/alexnet/Conv5_concept_max'
    path2 = 'result/wrong_csv_alblation/zero_out/alexnet-r/Conv5_concept_max'
    list1 = os.listdir(path1)
    list2 = os.listdir(path2)
    inter_table = []
    filename = []
    for file_name in list1:
        concept = file_name.split('_')
        concept.pop(0)
        concept.pop(1)
        concept = '_'.join(concept)
        filename.append(file_name)
        if concept in labels:
            inter_table.append(True)
        else:
            inter_table.append(False)
    inter_table = pd.DataFrame(inter_table, columns=['Intersect'])
    inter_table['File'] = filename
    for idx, row in inter_table.iterrows():
        if not row.Intersect:
            os.remove(f'{path1}/{row.File}')
            os.remove(f'{path2}/{row.File}')

def moveFilesOut(folder, out_folder):
    sub_folders = os.listdir(folder)
    for sub_folder in sub_folders:
        files = os.listdir(f'{folder}/{sub_folder}')
        for file_name in files:
            shutil.move(f'{folder}/{sub_folder}/{file_name}', out_folder)

def find_intersection_labels(file1, file2, out_file, by=None):
    if by is None:
        layer_data_1 = pd.read_csv(file1)
        layer_data_2 = pd.read_csv(file2)
        labels1 = layer_data_1.label
        labels2 = layer_data_2.label
        labels = pd.merge(labels1, labels2, how='inner')
        labels = labels.label.unique()
        pd.DataFrame(labels, columns=['Concept']).to_csv(out_file)
    elif by == 'category':
        pass


# create intersect dataste with imagenet layout
def createIntersectDataset(orgPath, out_path, correct_csv):
    correct_list = pd.read_csv(correct_csv).File.to_list()
    if orgPath is None:
        path_in_list = correct_list[0].split("/")[:-2]
        orgPath = '/'.join(path_in_list)
        full_path = True
    else:
        full_path = False
    imgnet_folders = os.listdir(orgPath)
    for folder in tqdm(imgnet_folders):
        class_folder = f'{out_path}/{folder}'
        os.mkdir(class_folder)
        img_list = os.listdir(f'{orgPath}/{folder}')
        for img in img_list:
            if full_path:
                img = f'{orgPath}/{folder}/{img}'
                if img in correct_list:
                    shutil.copy(f'{img}', class_folder)
            else:
                if img in correct_list:
                    shutil.copy(f'{orgPath}/{folder}/{img}', class_folder)


# plot ablation results when targeting on concepts
def abl_plot_concept(networks, alblation_type, layer_list, scale_list, plot_type, data_type="stander"):
    plot_name = {networks[0]: "", networks[1]: ""}
    for layer in layer_list:
        layer_name = layerNames(networks[0])
        if data_type == 'stander':
            specification = f'{layer_name[layer]}_concept_same'
        else:
            specification = f'{layer_name[layer]}_{data_type}_concept_same'
        by = 'concept'
        if alblation_type == 'keep':
            keep = True
        else:
            keep = False
        accuracy_stats = find_accuracy_stats(networks, specification, by, layer, keep=keep)
        # plot_type = 'max'
        for network in networks:
            concept_summary_path = f'result/tally/{network}_summary.csv'
            concept_summary = pd.read_csv(concept_summary_path, index_col=0)
            save_path = f'result/alblation_plots/{alblation_type}/{network}'
            img_name = plot_concept_importance(concept_summary, accuracy_stats, network, specification,
                                                     compared_layer=layer, out_dir=save_path, plotby=plot_type,
                                                     alblation_type=alblation_type, sortby='concept', scale=scale_list[layer])
            plot_name[network] = img_name
        layerwise_path = f'result/alblation_plots/{alblation_type}'
        os.system(f"montage -quiet {plot_name[networks[0]]} {plot_name[networks[1]]} -tile 1x2 -geometry +0+0 "
                  f"{layerwise_path}/{networks[0]}_{specification}_{plot_type}.jpg")
        os.system(f"rm {plot_name[networks[0]]} {plot_name[networks[1]]}")

# compute bb box range of the target images
def bb_range_count(file_path, full_path=True):
    target_file = pd.read_csv(file_path, index_col=0)
    img_name_list = []
    for idx, row in target_file.iterrows():
        img_name = row.File.split('/')[-1]
        if img_name[-3:] == 'png':
            img_name = f'{img_name[:-3]}JPEG'
        img_name_list.append(img_name)
    target_file.File = img_name_list

    # images in bb range
    bb_folders = os.listdir('imagenet_texture_bb')
    bb_folders.sort()
    bb_dict = {}
    bb_count = {}
    for folder in bb_folders:
        bb_dict[folder] = os.listdir(f'imagenet_texture_bb/{folder}')
        bb_count[folder] = 0

    # count bb range
    img_counter = 0
    bb_counter = 0
    for img in img_name_list:
        img_counter += 1
        for key, img_list in bb_dict.items():
            if img in img_list:
                bb_count[key] += 1
                bb_counter += 1
                continue

    print(f"miss count {img_counter - bb_counter}")
    return bb_count

def scrimble_img(null_img):
    patches = []
    grid_size = 112
    for ii in range(0, 229 - grid_size, grid_size):
        for jj in range(0, 229 - grid_size, grid_size):
            patches.append(null_img[:, :, ii:ii + grid_size, jj:jj + grid_size])
            np.random.seed(seed=212)
            randomize_patch = np.random.permutation(len(patches))
            patch_ind = 0
            texture_img = torch.zeros_like(null_img)
            for ii in range(0, 229 - grid_size, grid_size):
                for jj in range(0, 229 - grid_size, grid_size):
                    texture_img[0, :, ii:ii + grid_size, jj:jj + grid_size] = patches[randomize_patch[patch_ind]]
                    patch_ind += 1
    img = texture_img.clone()
    return img

def image_name_to_netid():
    # Map imagenet names to their netids
    input_f = open("ILSVRC2012/imagenet_validation_imagename_labels.txt")
    label_map = {}
    netid_map = {}
    for line in input_f:
        parts = line.strip().split(" ")
        label_map[parts[0]] = parts[1]
        netid_map[parts[1]] = parts[2]
    return label_map, netid_map

if __name__ == '__main__':
    '''Save intersect'''
    # intersect = np.load(f'/home/chirag/convergent_learning/interesection_resnet50_resnet50_r.npy')
    # target_path = 'dataset/correct/resnet50_inter'
    # check_path(target_path)
    # for img in intersect:
    #     img_path = f'ILSVRC2012/ILSVRC2012_img_val/{img}'
    #     shutil.copy(img_path, target_path)

    """ Rename by index"""
    # rename_path = 'dataset/wrong_csv_alblation/alexnet-r/Conv5'
    # rename_by_idx(rename_path)

    '''find stats for tally files'''
    # models = ['alexnet', 'alexnet-r', 'googlenet', 'googlenet-r', 'resnet50', 'resnet50-r']
    # for model in models:
    #     dissect_label_stats(f'result/tally/{model}', model)

    '''Copy intersect datas to form a new dataset'''
    # model = 'alexnet'
    # new_path = f'dataset/correct/{model}_inter'
    # check_path(new_path)
    # img_list = pd.read_csv(f'dataset/correct_csv/{model}_inter.csv', index_col=0).index.values.tolist()
    # copy_img(img_list, new_path)

    ''' Plot from surgery csv files'''
    # networks = ['alexnet', 'alexnet-r']
    # topk = 20
    # for network in networks:
    #     file_path = f'result/observe_surgery/stats/{network}'
    #     save_path = f'result/observe_surgery/plots/{network}/orange_1'
    #     check_path(save_path)
    #     stat_files = os.listdir(file_path)
    #     for file_name in stat_files:
    #         plot_title = f'{network}  layer {file_name[:-4]}'
    #         plot_surgery_by_layer(f'{file_path}/{file_name}', topk, plot_title, f'{save_path}/{file_name[:-4].zfill(2)}.jpg')
    #     os.system(f'montage {save_path}/*.jpg -tile 4x1 -geometry +40+0 total.jpg')


    """ The following is alblation experiments """

    ''' Fix Naming issue'''
    # fix_name('alexnet-r', '11', 'result/correct_csv_alblation/keep/alexnet-r/Conv5', 'csv')


    '''Display images after alblation'''
    # alblation_type = 'zero_out'
    # networks = ['resnet50', 'resnet50-r']
    # specification = 'chequered_stylized_same'
    # show_images_after_alblation(alblation_type, networks, specification, False, full_path=True, data_type='stylized')

    ''' Plot histogram of accuracy after ablation --- zero out by concepts'''
    # networks = ['alexnet', 'alexnet-r']
    # alblation_type = 'zero_out'
    # layer_list = ['1', '4', '7', '9', '11']
    # # normal imagenet scale
    # scale_list = {'1': (0, 50), '4': (0, 25), '7': (0, 12), '9': (0, 10), '11': (0, 5)}
    # # Gaussian imagenet scale
    # # scale_list = {'1': (0, 74), '4': (0, 42), '7': (0, 30), '9': (0, 30), '11': (0, 30)}
    # plot_type = 'max'
    # data_type = "stander"
    # abl_plot_concept(networks, alblation_type, layer_list, scale_list, plot_type, data_type)

    ''' Plot histogram of accuracy after ablation --- zero out by channels'''



    '''Find intersect concepts in each layer '''
    # network = ['resnet50', 'resnet50-r']
    # out_dir = f'result/tally/inter_concepts'
    # check_path(out_dir)
    # for layer in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']:
    #     find_intersection_labels(f'result/tally/{network[0]}/tally{layer}.csv', f'result/tally/{network[1]}/tally{layer}.csv', f'result/tally/inter_concepts/{network[0]}_{network[1]}_{layer}.csv')

    ''' create intetsect dataset'''
    # data_path = ['dataset/gaussian_noise', '/home/chirag/convergent_learning/scrambling_dataset_112/', '/home/chirag/stylized_imagenet/val']
    # for network in ['alexnet']:
    #     for version in ['gaussian', 'scramble', 'stylized']:
    #         # find inter
    #         inter_data = find_intersect(f'dataset/correct_csv/{network}_{version}.csv', f'dataset/correct_csv/{network}-r_{version}.csv')
    #         inter_data.to_csv(f'dataset/correct_csv/{network}_{version}_inter.csv')
    #
    #         target = f'{network}_{version}_inter'
    #         # org_dataset = '/home/chirag/convergent_learning/scrambling_dataset_112/'
    #         out_dir = f'dataset/correct/{target}'
    #         check_path(out_dir)
    #         csv = f'dataset/correct_csv/{target}.csv'
    #         createIntersectDataset(None, out_dir, csv)



    ''' Count bonding box range'''
    # target_path = 'result/wrong_csv_alblation/zero_out/resnet50'
    # tested_concepts = ['chequered_same', 'chequered_stylized_same', 'striped_same', 'striped_stylized_same', 'zigzagged_same', 'zigzagged_stylized_same']
    # bb_range_frame = pd.DataFrame(index=['0.0_0.1', '0.1_0.2', '0.2_0.3', '0.3_0.4', '0.4_0.5', '0.5_0.6', '0.6_0.7', '0.7_0.8', '0.8_0.9', '0.9_1.0'], columns=tested_concepts)
    # for folder in tested_concepts:
    #     print(f'Counting {folder}:')
    #     file_name = os.listdir(f'{target_path}/{folder}')[0]
    #     target_file = f'{target_path}/{folder}/{file_name}'
    #     bb_range_dict = bb_range_count(target_file)
    #     bb_range_frame[folder] = pd.DataFrame(bb_range_dict, index=[folder]).values[0]
    # out_dir = 'result/count_bb_range/zero_out/resnet50'
    # check_path(out_dir)
    # name = 'three_special_concepts.csv'
    # out_file = f"{out_dir}/{name}"
    # bb_range_frame.T.to_csv(out_file)

    """ find accuracy after ablation"""
    # networks = ['resnet50', 'resnet50-r']
    # surgery = 'org'
    # org_result = f'dataset/correct_csv/{networks[0]}_inter.csv'
    # for network in networks:
    #     out_dir = f'result/zero_out_acc{network}'
    #     target_folder = f'result/wrong_csv_alblation/zero_out/{network}/concepts_{surgery}'
    #     acc_stats = find_accuracy(target_folder, org_result, by='concept')
    #     # filter non-intersect conepts
    #     acc_stats = acc_stats[~np.array(acc_stats.Counts == 0)]
    #     acc_stats.sort_values(by='Counts', inplace=True, ascending=False)
    #     acc_stats.to_csv(f'{out_dir}/{network}_{surgery}.csv')

    """Acc drop"""
    # surgery_list = ['org', 'stylized', 'scramble']
    # networks = ['resnet', 'resnet-r']
    # stats = []
    # column = []
    # for surgery in surgery_list:
    #     for network in networks:
    #         stats.append(pd.read_csv(f'result/zero_out_acc/{network}_{surgery}.csv', index_col=0))
    #         column.append(f'{network}_{surgery}')
    #
    # all_in_one = pd.DataFrame(index=stats[0].index, columns=column)
    # for data, col_name in zip(stats, column):
    #     all_in_one[col_name] = data.Accuracy_Drop*100
    # all_in_one['Count'] = data.Counts
    # all_in_one.set_index(stats[0].Target, drop=True, inplace=True)
    # all_in_one.round(decimals=2).to_csv('result/zero_out_acc/resnet_acc_drop.csv')

    ''' Acc drop layer-wise Alexnet'''
    # networks = ['alexnet', 'alexnet-r']
    # datasets = ['org', 'gaussian', 'scramble', 'stylized']
    # titles = ['Clean ImageNet', 'Gaussian ImageNet', 'Scrambled ImageNet', 'Stylized ImageNet']
    # # datasets = ['gaussian', 'scramble', 'stylized']
    # # datasets = ['org', 'gaussian']
    # target_path = 'result/wrong_csv_alblation/zero_out'
    # out_path = 'result/alblation_plots/zero_out'
    # compare_path = 'dataset/correct_csv'
    # # layers = ['Conv2', 'Conv3', 'Conv4', 'Conv5']
    # layers = ['Conv5']
    # scale = None
    #
    # for layer in layers:
    #     all_stats = []
    #     for data_type in datasets:
    #         network_acc_stats = []
    #         target_folder = f'{layer}_{data_type}_concept_same'
    #         for network in networks:
    #             layer_data = find_accuracy(f'{target_path}/{network}/{target_folder}', f'{compare_path}/alexnet_{data_type}_inter.csv', by='concept').sort_values(by='Target')
    #             network_acc_stats.append(layer_data)
    #         all_stats.append(network_acc_stats)
    #
    #     # sort according to R net
    #     stander = all_stats[0][1].sort_values(by='Accuracy_Drop', ascending=False)
    #     stander.index = stander.Target
    #     sorted_all = []
    #     for data_pair in all_stats:
    #         sorted_pair = []
    #         for data in data_pair:
    #             data.index = data.Target
    #             _, sorted_data = stander.align(data, join='left')
    #             sorted_data.reset_index(drop=True, inplace=True)
    #             sorted_pair.append(sorted_data)
    #         sorted_all.append(sorted_pair)
    #
    #     # single layer plot
    #     global_stats = []
    #     for idx, pair_stats in enumerate(sorted_all):
    #         stats = []
    #         concepts = pair_stats[0].Target.to_list()
    #         xtick = []
    #         for concept in concepts:
    #             name = '_'.join(concept.split('_')[1:])
    #             name = '-'.join(name.split('-')[:1])
    #             xtick.append(name)
    #         out_dir = f'{out_path}/{networks[0]}'
    #         check_path(out_dir)
    #         # plot_name = f'Ablation_{layer}_{str(idx).zfill(2)}{datasets[idx]}.pdf'
    #         plot_name = f'Ablation_{layer}_{datasets[idx]}.pdf'
    #         stats.append(pair_stats[0].Accuracy_Drop.values*100)
    #         stats.append(pair_stats[1].Accuracy_Drop.values*100)
    #         global_stats.append(stats)
    #
    #         scatterPlot(stats, xtick, out_dir, plot_name, img_size='auto', pltheight=3, fontsize=12, scale=scale)

    '''Global accuracy drop plot '''
    # networks = ['alexnet', 'alexnet-r']
    # datasets = ['org', 'gaussian', 'scramble', 'stylized']
    # # datasets = ['gaussian', 'scramble', 'stylized']
    # # datasets = ['org']
    # target_path = 'result/correct_csv_alblation/zero_out'
    # out_path = 'result/alblation_plots/zero_out'
    # compare_path = 'dataset/correct_csv'
    #
    #
    # all_stats = []
    # for data_type in datasets:
    #     network_acc_stats = []
    #     # target_folder = f'{data_type}_concept_same_orgSet'
    #     target_folder = f'Conv5_concept_same_orgSet'
    #     for network in networks:
    #         # for intersent
    #         # layer_data = find_accuracy(f'{target_path}/{network}/{target_folder}', f'{compare_path}/alexnet_{data_type}_inter.csv', by='concept').sort_values(by='Target')
    #         # for org dataset
    #         layer_data = find_accuracy(f'{target_path}/{network}/{target_folder}',
    #                                    f'{compare_path}/{network}.csv', by='concept', orgSet=True).sort_values(by='Target')
    #         network_acc_stats.append(layer_data)
    #     all_stats.append(network_acc_stats)
    #
    # # sort according to R net
    # stander = all_stats[0][1].sort_values(by='Accuracy_Drop', ascending=False)
    # stander.index = stander.Target
    # sorted_all = []
    # for data_pair in all_stats:
    #     sorted_pair = []
    #     for data in data_pair:
    #         data.index = data.Target
    #         _, sorted_data = stander.align(data, join='left')
    #         sorted_data.reset_index(drop=True, inplace=True)
    #         sorted_pair.append(sorted_data)
    #     sorted_all.append(sorted_pair)
    #
    # # single layer plot
    # global_stats = []
    # for idx, pair_stats in enumerate(sorted_all):
    #     stats = []
    #     concepts = pair_stats[0].Target.to_list()
    #     xtick = concepts
    #     # for concept in concepts:
    #     #     xtick.append('_'.join(concept.split('_')[1:]))
    #     out_dir = f'{out_path}/{networks[0]}'
    #     check_path(out_dir)
    #     plot_name = f'Ablation_Conv5_{datasets[idx]}_50k.pdf'
    #     stats.append(pair_stats[0].Accuracy_Drop.values*100)
    #     stats.append(pair_stats[1].Accuracy_Drop.values*100)
    #     global_stats.append(stats)
    #     scatterPlot(stats, xtick, out_dir, plot_name, title=datasets[idx])

    # plot all dataset together
    # xtick = datasets
    # out_dir = f'{out_path}/{networks[0]}'
    # check_path(out_dir)
    # plot_name = f'Conv5_all_dataset.pdf'
    # scatterPlot(global_stats, xtick, out_dir, plot_name, img_size=(6, 4))

    ''' Acc drop layer-wise all in one'''
    # networks = ['alexnet', 'alexnet-r']
    # ablation_type = 'keep'
    # datasets = ['org', 'gaussian', 'scramble', 'stylized']
    # titles = ['Clean ImageNet', 'Gaussian ImageNet', 'Scrambled ImageNet', 'Stylized ImageNet']
    # target_path = 'result/wrong_csv_alblation/zero_out'
    # out_path = 'result/alblation_plots/zero_out'
    # compare_path = 'dataset/correct_csv'
    # layers = ['Conv5']
    #
    #
    # for layer in layers:
    #     all_stats = []
    #     for data_type in datasets:
    #         network_acc_stats = []
    #         target_folder = f'{layer}_{data_type}_concept_same'
    #         for network in networks:
    #             layer_data = find_accuracy(f'{target_path}/{network}/{target_folder}', f'{compare_path}/alexnet_{data_type}_inter.csv', by='concept').sort_values(by='Target')
    #             network_acc_stats.append(layer_data)
    #         all_stats.append(network_acc_stats)
    #
    #     # sort according to R net
    #     stander = all_stats[0][1].sort_values(by='Accuracy_Drop', ascending=False)
    #     stander.index = stander.Target
    #     sorted_all = []
    #     for data_pair in all_stats:
    #         sorted_pair = []
    #         for data in data_pair:
    #             data.index = data.Target
    #             _, sorted_data = stander.align(data, join='left')
    #             sorted_data.reset_index(drop=True, inplace=True)
    #             sorted_pair.append(sorted_data)
    #         sorted_all.append(sorted_pair)
    #
    #     # use same xtick for all plots
    #     xtick = []
    #     out_dir = f'{out_path}/{networks[0]}'
    #     for concept in sorted_all[0][0].Target.to_list():
    #         name = '_'.join(concept.split('_')[1:])
    #         name = '-'.join(name.split('-')[:1])
    #         xtick.append(name)
    #     # single layer plot for all dataset
    #     stats = []
    #     plot_name = f'Ablation_{layer}_4_dataset.pdf'
    #     scale = (0, 33)
    #     for idx, pair_stats in enumerate(sorted_all):
    #         stats.append([pair_stats[0].Accuracy_Drop.values*100, pair_stats[1].Accuracy_Drop.values*100])
    #     scatterPlot2(stats, xtick, out_dir, plot_name, img_size=(12,4), pltheight=4, fontsize=12, scale=scale, topk=10, title=None)

    ''' Find intersect images from clean and noisy dataset'''
    # out_path = 'dataset/correct/alexnet_noise_super_inter'
    # check_path(out_path)
    # csv_file = 'dataset/correct_csv/alexnet_clean_noise_inter_noisepath.csv'
    # createIntersectDataset(None, out_path, csv_file)

    ''' Montage channel acitvations'''
    # result_path = 'result/channel_activation'
    # networks = ['alexnet', 'alexnet-r']
    # layers = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5']
    # data_type = ['clean', 'noise']
    # # for network in networks:
    # #     for img in range(10):
    # #         for layer in layers:
    # #             type_plot_list = []
    # #             for d_type in data_type:
    # #                 type_plot_name = f"{result_path}/{network}/img{img:02d}_{layer}_{d_type}.jpg"
    # #                 os.system(f"montage -quiet {result_path}/{network}/{d_type}/img{img:02d}_{layer}*.jpg -tile 16x -geometry +1+1 {type_plot_name}")
    # #                 type_plot_list.append(type_plot_name)
    # #             # montage clean and noise version side by side
    # #             side_by_side_plot_name = f"{result_path}/{network}/img{img:02d}_{layer}.jpg"
    # #             os.system(f"montage -quiet {type_plot_list[0]} {type_plot_list[1]} -tile 1x2 -geometry +0+10 {side_by_side_plot_name}")
    # #             os.system(f"rm {type_plot_list[0]} {type_plot_list[1]}")
    #
    # # montage with origin image
    # for i in range(10):
    #     for layer in layers:
    #         os.system(f"montage {result_path}/org_img/img_org_{i:02d}.jpg "
    #                   f"{result_path}/{networks[0]}/img{i:02d}_{layer}.jpg "
    #                   f"{result_path}/{networks[1]}/img{i:02d}_{layer}.jpg -tile 3x1 -geometry +5+0 "
    #                   f"{result_path}/result/{layer}_img{i:02d}.jpg")

    # path = 'result/channel_activation/clean_img'
    # out_path = 'result/channel_activation/clean_img_resize'
    # check_path(out_path)
    # files = os.listdir(path)
    # for file in files:
    #     full_path = os.path.join(path, file)
    #     img = Image.open(full_path)
    #     width, height = img.size
    #     if width >= height:
    #         size = (int(width*(256/height)), 256)
    #     else:
    #         size = (256, int(256/width)*height)
    #     img = img.resize(size)
    #     width, height = img.size
    #     img = img.crop(((width-224)/2, (height-224)/2, (width+224)/2, (height+224)/2))
    #     out_full_path = os.path.join(out_path, file)
    #     img.save(out_full_path)


    # tested = ['alpha_100M', 'alpha_1B', 'alpha_10B', 'alpha_100B']
    # model = 'ResNet18'
    # path = 'result/grad_data'
    # out_path = 'result/grad_img'


    # for test in tested:
    #     file_path = f'{path}/{model}_{test}'
    #     img_path = f'{out_path}/{model}_{test}'
    #     plot_grad(file_path, img_path)

    # montage results
    # montage_tv_train_grad(tested)

    ''' Find intersection of correctly classified images'''
    # net1 = 'googlenet'
    # net2 = 'googlenet-r'
    # snet = np.load(f'data/{net1}.npy')
    # rnet = np.load(f'data/{net2}.npy')
    # intersect = np.intersect1d(snet, rnet)
    # print(len(intersect))
    # np.save(f'data/intersection_{net1}_{net2}.npy', intersect)

    ''' Copy images to according to ImageNet-CL'''
    # networks = ['alexnet', 'googlenet', 'resnet50']
    # for network in networks:
    #     file_name = f'intersection_{network}_{network}-r.npy'
    #     save_path = f'dataset/{network}_silhouette'
    #     os.makedirs(save_path, exist_ok=True)
    #     image_list = np.load(f'data/{file_name}')
    #     silouette_path = 'dataset/silhouette_imagenet_val_opencv'
    #     for image_name in image_list:
    #         shutil.copyfile(f'{silouette_path}/{image_name}', f'{save_path}/{image_name}')


    ''' Plot network Mean TV'''
    # data_path = 'result/model_weights_TV'
    # model_pairs = [['AlexNet', 'AlexNet-R'], ['GoogLeNet', 'GoogLeNet-R'], ['ResNet50', 'ResNet50-R']]
    # for models in model_pairs:
    #     data_s = pd.read_csv(f'{data_path}/{str.lower(models[0])}.csv', index_col=0)
    #     data_r = pd.read_csv(f'{data_path}/{str.lower(models[1])}.csv', index_col=0)
    #     stats = [round(data_s.Mean_TV, 2), round(data_r.Mean_TV, 2)]    
    #     scatterPlotPair(stats, xtick=data_s.index, out_path=data_path, plot_name=f'{models[0]}.pdf', img_size=(8,4), ylabel='Mean TV', networks=[models[0], models[1]], title=None, pltheight=4, fontsize=14, scale=None)
        # # compute network wise mean and report
        # datas = [data_s, data_r]
        # for model, data in zip(models, datas):
        #     cumulated_tv = 0
        #     channel_sum = 0
        #     for index, row in data.iterrows():
        #         import ipdb
        #         # ipdb.set_trace()
        #         cumulated_tv += row.Mean_TV * row.Channels
        #         channel_sum += row.Channels
        #     mean_tv = cumulated_tv/channel_sum
        #     print(f'{model} mean TV: {mean_tv}')


    ''' Image 4A (Real-scrambled-stylized-contour-silhouette)'''
    # real_data = '/home/peijie/ILSVRC2012/val'
    # scramble_data = '/home/chirag/convergent_learning/scrambling_dataset_112'
    # stylized_data = 'dataset/stylized_imagenet_val2'
    # contour_data = 'dataset/silhouette_imagenet_val'
    # silhouette_data = '/home/chirag/gpu3_codes/deeplab-pytorch/imagenet_shape_dataset'
    # label_map, netid_map = image_name_to_netid()

    # out_path = 'result/figure4A'
    # org_plot_images = 'result/figure4A/source_image'
    # os.makedirs(out_path, exist_ok=True)
    # os.makedirs(org_plot_images, exist_ok=True)

    # for idx in range(5):
    #     # choose an image from silthoette
    #     for each_class in os.listdir(silhouette_data):
    #         image_list = os.listdir(f'{silhouette_data}/{each_class}')
    #         # randomly pick one image from each class
    #         image = random.choice(image_list)
    #         silhouette_image_path = f'{silhouette_data}/{each_class}/{image}'
    #         netid = netid_map[label_map[image]] # get netid so that we could find the images in val set

    #         # copy images to a folder
    #         shutil.copyfile(silhouette_image_path, f'{org_plot_images}/temp_silhouette_{each_class}.JPEG')
    #         shutil.copyfile(f'{contour_data}/{image}', f'{org_plot_images}/temp_contour_{each_class}.JPEG')
    #         shutil.copyfile(f"{stylized_data}/{netid}/{image.split('.')[0]}.png", f'{org_plot_images}/temp_tylized_{each_class}.png')
    #         shutil.copyfile(f'{scramble_data}/{netid}/{image}', f'{org_plot_images}/temp_scramble_{each_class}.JPEG')
    #         shutil.copyfile(f'{real_data}/{netid}/{image}', f'{org_plot_images}/temp_real_{each_class}.JPEG')
    #         # resize image
    #         for img_tpye in ['real', 'silhouette']:
    #             temp = Image.open(f"{org_plot_images}/temp_{img_tpye}_{each_class}.JPEG")
    #             temp = temp.resize((224,224))
    #             if temp.mode in ("RGBA", "P"): temp = temp.convert("RGB")
    #             temp.save(f"{org_plot_images}/temp_{img_tpye}_{each_class}.JPEG")
    #         temp = Image.open(f"{org_plot_images}/temp_contour_{each_class}.JPEG")
    #         temp = temp.resize((224,224))
    #         temp.save(f"{org_plot_images}/temp_contour_{each_class}.png")
    #         # montage all images
    #         os.system(f"montage -quiet {org_plot_images}/temp_real_{each_class}.JPEG {org_plot_images}/temp_scramble_{each_class}.JPEG {org_plot_images}/temp_tylized_{each_class}.png \
    #             {org_plot_images}/temp_contour_{each_class}.png {org_plot_images}/temp_silhouette_{each_class}.JPEG -geometry +0+0 -tile x1 {out_path}/{each_class}_{idx:02}.JPEG")
    #         os.system(f'rm -rf {org_plot_images}/temp*')


    # montage images
    # data_path = 'result/figure4A'
    # os.system(f"montage {data_path}/airplane_03.JPEG {data_path}/bear_04.JPEG {data_path}/bicycle_01.JPEG {data_path}/bird_03.JPEG \
    #     {data_path}/boat_03.JPEG {data_path}/bottle_02.JPEG {data_path}/car_00.JPEG -geometry +0+2 -tile x10 {data_path}/formal_figure4A.JPEG")

    ''' Formal figure 1'''
    # data_path = 'formal_figure_1'
    # order = ['real', 'adversarial', 'gaussian', 'scrambled', 'stylized', 'contour', 'silhouette', ]
    # # resize
    # for idx, folder in enumerate(order):
    #     img_file = os.listdir(f"{data_path}/{folder}")[0]
    #     temp = Image.open(f"{data_path}/{folder}/{img_file}")
    #     temp = temp.resize((224,224))
    #     temp.save(f"{data_path}/{idx:02}_{img_file}")
    # sys.exit(0)

    ''' Compute mCE '''
    
    alexnet_data_path = 'result/imagenet_c_stats/alexnet'
    resnet_data_path = 'result/imagenet_c_stats/resnet'
    alexnet_file_list = os.listdir(alexnet_data_path)
    alexnet_file_list.sort()
    resnet_file_list = os.listdir(resnet_data_path)
    resnet_file_list.sort()
    for alexnet_file, resnet_file in zip(alexnet_file_list, resnet_file_list):
        if alexnet_file.split('_', 2)[-1] != resnet_file.split('_', 2)[-1]:
            print('Data match fail..')
            os.exist(0)
        alexnet_stats = pd.read_csv(f'{alexnet_data_path}/{alexnet_file}', index_col=0)
        resnet_stats = pd.read_csv(f'{resnet_data_path}/{resnet_file}', index_col=0)
        # ipdb.set_trace()
        ce = (100 - resnet_stats).sum().values[0]/(100 - alexnet_stats).sum().values[0]
        distortion = alexnet_stats.columns[0]
        # ipdb.set_trace()
        print(f'CE of {distortion}: {round(ce*100, 3)}%')
import numpy as np
import pandas as pd
import os
import sys, getopt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import cm
import glob
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage import feature
plt.style.use('ggplot')

def autolabel(rects, dis_tpye='int', keep=3, text_float=0.05):
    for rect in rects:
        height = rect.get_height()
        if dis_tpye == 'int':
            dis_value = abs(int(height))
        elif dis_tpye == 'float':
            dis_value = abs(round(height, keep))
        plt.text(rect.get_x(), height+text_float, f'{dis_value}', fontsize=9)

class PlotTool:

    def __init__(self, path_to_csv_folder, model_Name, category=None, scale=None, useFiles=True):
        # path and model names
        self.path        = path_to_csv_folder
        self.model_Name  = model_Name
        self.scale       = scale
        self.stats       = None
        # create stat folder
        self.stats_folder = os.path.join(path_to_csv_folder,'stats')
        if not os.path.isdir(self.stats_folder):
            os.mkdir(self.stats_folder)  
        # find csv file names
        if useFiles:
            self.files = self.getCSV(self.path)

        # category setting      
        self.category = category
        self.fix_cate = True
        if self.category == None:
            self.fix_cate = False


    # get csv_name file name list
    def getCSV(self, path):
        current_path = os.getcwd()
        os.chdir(path) 
        files = glob.glob("*.csv")
        os.chdir(current_path)
        return files

    # read csv_name file and return categroy and score stats
    def findStats(self, filePath):
        rawData = pd.read_csv(filePath)
        # use fix category
        categories = self.category
        if not self.fix_cate:
            # vary in category
            categories = list(rawData.category.unique())
        categories.sort()
        data_category = pd.DataFrame(np.zeros((1,len(categories)),dtype=np.int8), columns=categories ,index=['Number of Units'], dtype=np.int16)
        iou_array = np.array([], dtype=np.float16)
        for i, cate in rawData.category.iteritems():
            if cate in categories:
                data_category.iloc[0][cate] += 1
                iou_array = np.append(iou_array, rawData.loc[i]['score'])
        return [data_category, iou_array], categories

    # layer names of dissection
    def layerNames(self):
        model = str.lower(self.model_Name)
        if model in  ['alexnet', 'alexnet-r']:
            layer_list = {'1':'conv1', '4':'conv2', '7':'conv3', '9':'conv4', '11':'conv5'}

        elif model in ['googlenet', 'googlenet-r']:
            layers = ["conv1", "conv2", "conv3", "inception3a", "inception3b", "inception4a", "inception4b",
                    "inception4c", "inception4d", "inception4e", "inception5a", "inception5b"]
            layer_list = dict(zip(layers, layers))
        elif model in ['resnet50', 'resnet50-r']:
            layers = ['bn1', 'conv1', 'layer1', 'layer2', 'layer3', 'layer4']
            layer_list = dict(zip(layers, layers))
        else:
            print("Network ",self.model_Name ," unsupported yet. Please update 'layerNames()' function.")
            os._exit(0)         
        return layer_list

    # bar plot and save images as png file in folder 'stats'
    def plotLayer(self, files, save=True):
        layer_name = self.layerNames()
        stats = []
        scale_record = {}
        for csv_name in files:
            dissect_stats, categories = self.findStats(os.path.join(self.path, csv_name))
            stats.append(dissect_stats)
            # statistics for plot
            values = dissect_stats[0].to_numpy()[0]
            mean = round(dissect_stats[1].mean(), 3)
            median = round(np.median(dissect_stats[1]), 3)
            # plot 
            mycm = cm.get_cmap('Set3', values.size)
            rect=plt.bar(np.arange(values.size), values, bottom=0, color=mycm.colors)
            plt.ylabel('Number of Units')
            if self.scale is None:
                plt.autoscale(axis='y')
            else:
                plt.ylim(self.scale[csv_name])
            plt.title(self.model_Name+" Mean, Median IoU : "+str(mean)+', '+str(median))
            plt.xticks(np.arange(values.size),categories)
            plt.legend(rect,categories, loc=0)
            plt.rcParams['figure.figsize'] = [6,4]
            # save plot
            if save:
                plt.savefig(os.path.join(self.stats_folder, layer_name[csv_name[5:-4]])+"_cate.jpg")
            # update scales
            scale_record[csv_name] = plt.ylim()
            plt.close('all')

        return scale_record, stats


    # def plotFromStats(self, stats, name='Global stats', scale=None, save=True):
    #     # stats for plot
    #     values = stats[0].to_numpy()[0]
    #     mean = round(stats[1].mean(), 3)
    #     median = round(np.median(stats[1]), 3)
    #     # plot 
    #     mycm = cm.get_cmap('Set3', values.size) # choose color
    #     rect=plt.bar(np.arange(values.size), values, bottom=0, color=mycm.colors)
    #     plt.ylabel('Number of Units')
    #     if self.scale is None:
    #         plt.autoscale(axis='y')
    #     else:
    #         plt.ylim(self.scale)
    #     plt.title(self.model_Name+" Mean, Median IoU : "+str(mean)+', '+str(median))
    #     plt.xticks(np.arange(values.size),self.category)
    #     plt.legend(rect,self.category, loc=0)
    #     plt.rcParams['figure.figsize'] = [6,4]
    #     # save plot
    #     if save:
    #         plt.savefig(os.path.join(self.stats_folder, name+".png"))
    #     plt.close('all')
    
    # find stats for single categroy
    def categoryStats(self, category, sort='alphabet'):
        # store all texture units in a list respect to layer
        for csvfile in self.files:
            rawData = pd.read_csv(os.path.join(self.path,csvfile))
            rawData = rawData[rawData['category']==category]
            columns = rawData['label'].unique()
            frame = pd.DataFrame(np.zeros((1,len(columns))), index=['Number of Units'], columns=columns, dtype=np.int64)
            for _,row in rawData.iterrows():
                frame[row.label] += 1
            try:
                stats = pd.concat([stats, frame], axis=1)
            except:
                stats = frame

        columns = stats.sort_index(axis=1).columns.unique()
        result  = pd.DataFrame(np.zeros((1,len(columns))), index=['Number of Units'], columns=columns, dtype=np.int64)
        for label in columns:
            if isinstance(stats[label].sum(), np.int64):
                result[label] = stats[label].sum()
            else:
                result[label] = stats[label].sum(axis=1)[0]
        return result


def layerNames(model):
    if model in  ['alexnet', 'alexnet-r']:
        layer_list = {'1':'Conv1', '4':'Conv2', '7':'Conv3', '9':'Conv4', '11':'Conv5'}

    elif model in ['googlenet', 'googlenet-r']:
        layers = ["conv1", "conv2", "conv3", "inception3a", "inception3b", "inception4a", "inception4b",
                "inception4c", "inception4d", "inception4e", "inception5a", "inception5b"]
        layer_list = dict(zip(layers, layers))
    elif model in ['resnet50', 'resnet50-r', 'resnet50-sin', 'resnet50-sin+in', 'ResNet50', 'ResNet50-R', 
                        'resnet', 'resnet-r', 'resnet-sin', 'resnet-sin+in', 'ResNet', 'ResNet-R',
                        'ResNet-Sin', 'ResNet-SIN+IN']:
        layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        layer_list = dict(zip(layers, layers))
    else:
        print("Network ",model ," unsupported yet. Please update 'layerNames()' function.")
        os._exit(0)
    return layer_list

#plot single category
def plotCategory(stats, models, cate_name, stats_folder):
    scaleDisct = {}
    scale = 0
    # find scales
    for data, model in zip(stats, models):
        # statistics for plot
        values = data.to_numpy()[0]
        # plot 
        mycm = cm.get_cmap('plasma', values.size)
        # rect=plt.bar(np.arange(values.size), values, bottom=0, color=mycm.colors)
        rect=plt.bar(np.arange(values.size), values, bottom=0, color=mycm.colors)
        plt.ylabel('Number of Units')
        plt.autoscale(axis='y')

        # plt.title(model+cate_name)
        # plt.xticks(np.arange(values.size),data.columns)
        # # plt.legend(rect,categories, loc=0)

        # update scales
        scaleDisct[model] = plt.ylim()
        plt.close('all')

    for _, value in scaleDisct.items():
        scale = max(scale, max(value))

    # Align same labels for the data
    unique_labels = [len(stats[0].columns), len(stats[1].columns)]
    x_axis        = max(unique_labels)
    stats0, stats1 = stats[0].align(stats[1], join='inner', axis=1)
    stats0 = pd.merge(stats0, stats[0], how='left')
    stats1 = pd.merge(stats1, stats[1], how='left')
    stats = [stats0, stats1]



    # plot
    for data, model, path, label in zip(stats, models, stats_folder, unique_labels):
        # statistics for plot
        values = data.to_numpy()[0]
        # plot 
        mycm = cm.get_cmap('plasma', values.size)
        plt.rcParams['figure.figsize'] = [16,4]
        # rect=plt.bar(np.arange(values.size), values, bottom=0, color=mycm.colors)
        rect=plt.bar(np.arange(values.size), values, bottom=0, width=0.8, color=mycm.colors)
        plt.ylabel('Number of Units')
        if scale == 0:
            plt.autoscale(axis='y')
        else:
            plt.ylim(0,scale)
        plt.xlim(-1,x_axis)
        plt.title(model+' '+cate_name+': '+str(label)+' unique labels')
        # plt.xticks(np.arange(values.size),data.columns, rotation=40)
        plt.xticks(np.arange(x_axis),data.columns, rotation=40)
        autolabel(rect)
        # save plot
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'stats')+"/"+cate_name+".jpg")
        plt.close('all')

    return stats

def getFiles(path, fileType):
    current_path = os.getcwd()
    os.chdir(path) 
    files = glob.glob("*."+fileType)
    os.chdir(current_path)
    return files

# bar plot direct from stats
def plotFromStats(stats, models, paths, category):
    scaleDisct = {}
    scale = 0
    # find scales
    for data, model in zip(stats, models):
        values = data[0].to_numpy()[0]
        mycm = cm.get_cmap('Set3', values.size) # choose color
        rect=plt.bar(np.arange(values.size), values, bottom=0, color=mycm.colors)
        plt.ylabel('Number of Units')
        plt.autoscale(axis='y')
        # update scales
        scaleDisct[model] = plt.ylim()
        plt.close('all')

    for _, value in scaleDisct.items():
        scale = max(scale, max(value))

    for data, model, path in zip(stats, models, paths):
        out_dir = path+"/stats"
        # stats for plot
        values = data[0].to_numpy()[0]
        mean = round(data[1].mean(), 3)
        median = round(np.median(data[1]), 3)
        # plot 
        mycm = cm.get_cmap('Set3', values.size) # choose color
        rect=plt.bar(np.arange(values.size), values, bottom=0, color=mycm.colors)
        plt.ylabel('Number of Units')
        plt.autoscale(axis='y')
        plt.ylim(0,scale)
        plt.title(model+" Mean, Median IoU : "+str(mean)+', '+str(median))
        plt.xticks(np.arange(values.size), category)
        plt.legend(rect, category, loc=0)
        plt.rcParams['figure.figsize'] = [6,4]
        autolabel(rect)
        # save plot
        plt.savefig(os.path.join(out_dir, "global.jpg"))
        plt.close('all')
    
# plot difference between stats
def plotDiff(stats, models, out_dir, cate_name):
    stats0, stats1 = stats[0].align(stats[1], join='outer', axis=1, fill_value=0)
    diff = stats0-stats1
    diff.sort_values(by=0, axis=1, inplace=True, ascending=False)

    # plot difference
    values = diff.to_numpy()[0]
    mycm = cm.get_cmap('plasma', values.size)
    rect = plt.bar(np.arange(values.size), values, color=mycm.colors)
    plt.autoscale(axis='y')
    plt.xticks(np.arange(values.size),diff.columns, rotation=40)
    plt.title(cate_name+": "+models[0]+"(Top) vs. "+models[1]+"(Bottom)")
    autolabel(rect)
    plt.tight_layout()
    plt.savefig(out_dir+"/"+cate_name+"_diff.jpg")
    plt.close('all')

# get the brief class name
def brief_class(class_name):
    brief_name = []
    for name in class_name:
        brief_name.append(name.split(",", 1)[0])
    return brief_name

# plot the histogram for the validation results
def plot_val_hist(stats, title, path, class_name, top5=False):  
    values1 = stats.Top1.to_numpy(dtype=np.int16)
    values5 = stats.Top5.to_numpy(dtype=np.int16)

    # plot image and save
    mycm = cm.get_cmap('plasma', values1.size)
    rect = plt.bar(np.arange(values1.size), values1, color=mycm.colors)
    plt.autoscale(axis='y')
    plt.xticks(np.arange(values1.size),class_name, rotation=40)
    plt.title(title)
    autolabel(rect)
    plt.rcParams['figure.figsize'] = [16,4]
    plt.tight_layout()
    plt.savefig(path+"hist_top1.jpg")   
    plt.close('all') 

def val_diff_hist(path1, path2, result_path, threshold):
    stats1 = pd.read_csv(path1+"/hist.csv", index_col=0)
    stats1.pop('Class')
    stats2 = pd.read_csv(path2+"/hist.csv", index_col=0)
    stats2.pop('Class')
    model1 = path1.split("/")[-1]
    model2 = path2.split("/")[-1]


    # plot 1 is better than 2
    diff = stats1.apply(pd.to_numeric).Top1-stats2.apply(pd.to_numeric).Top1
    diff_sorted = diff.sort_values(ascending=False)
    diff_filterd = diff_sorted[:threshold]
    values = diff_filterd.to_numpy(dtype=np.int16)
    # plot image and save
    mycm = cm.get_cmap('plasma', values.size)
    plt.rcParams['figure.figsize'] = [16,4]
    rect = plt.bar(np.arange(values.size), values, color=mycm.colors)
    plt.autoscale(axis='y')
    plt.xticks(np.arange(values.size), diff_filterd.index.to_list(), rotation=40)
    plt.title(model1+"  better histrogram")
    autolabel(rect)
    plt.tight_layout()
    plt.savefig(result_path+"/"+model1+"_diff.jpg")   
    plt.close('all') 

    # plot 2 is better than 1
    diff = stats2.apply(pd.to_numeric).Top1-stats1.apply(pd.to_numeric).Top1
    diff_sorted = diff.sort_values(ascending=False)
    diff_filterd = diff_sorted[:threshold]
    values = diff_filterd.to_numpy(dtype=np.int16)
    # plot image and save
    mycm = cm.get_cmap('plasma', values.size)
    rect = plt.bar(np.arange(values.size), values, color=mycm.colors)
    plt.autoscale(axis='y')
    plt.xticks(np.arange(values.size), diff_filterd.index.to_list(), rotation=40)
    plt.title(model2+"  better histrogram")
    autolabel(rect)
    plt.rcParams['figure.figsize'] = [16,4]
    plt.tight_layout()
    plt.savefig(result_path+"/"+model2+"_diff.jpg")   
    plt.close('all')

def plot_surgery_by_layer(file_path, topk, title, save_path):
    stats = pd.read_csv(file_path, index_col=0)
    if isinstance(topk, int):
        stats = stats[:topk]
    values = stats.Score.to_numpy(dtype=np.float32)
    values = np.round(values, 1)
    labels = stats.Label.to_list()
    units  = stats.Unit.to_list()

    # bar plot for topk with labels
    mycm = cm.get_cmap('viridis', values.size)
    # plt.rcParams['figure.figsize'] = [16,4]
    rects = plt.bar(np.arange(values.size), values, color=mycm.colors)
    plt.autoscale(axis='y')
    plt.xticks(np.arange(values.size), labels, rotation=40)
    plt.title(title)
    # autolabel(rect)
    for idx, rect in enumerate(rects):
        height = rect.get_height()
        unit = units[idx]
        plt.text(rect.get_x(), height+0.05, '%s'%unit, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

def load_labelMap():
    # load lalbel name
    labelMap = pd.DataFrame(index=range(1000), columns=['CodeName', 'Name'])
    labels = {}
    names  = []
    with open("ILSVRC2012/synset_words.txt") as f:
        for line in f:
            codeName, name = line.strip().split(" ", 1)
            labels[codeName] = name
            names.append(name)

    labelMap.CodeName = labels.keys()
    labelMap.Name     = names
    return labelMap


def display_class_confidence_canny(model, class_name, rank, result_path, labelMap=None, display_number=10, topk=1):

    stats = pd.read_csv("validate/val_result/"+model+"/confidence.csv", index_col=0)
    if isinstance(class_name, int):
        index      = class_name
        confidence = stats.loc[index, 'Confidence']
        images     = stats.loc[index, 'Images']
    else:
        index = labelMap[labelMap['CodeName'].isin([class_name])].index[0]
        confidence = stats.loc[index, 'Confidence']
        images     = stats.loc[index, 'Images']

    data = pd.DataFrame(np.array([np.array(confidence.split(","), dtype=np.float32), images.split(",")]).T, columns=["Confidence", "Images"])
    sorted_data = data.sort_values(by='Confidence', ascending=False).reset_index(drop=True)

    if display_number == "all":
        display_number = len(sorted_data)

    plot_data = sorted_data[:display_number]
    imageList = plot_data.Images.to_list()
    confidence = plot_data.Confidence.to_list()

    font_size = 30
    new_im = Image.new('L', (2240,224+font_size))
    for i in range(1, display_number+1):
        image = Image.open("ILSVRC2012/ILSVRC2012_img_val/"+imageList[i-1]).convert('L')
        image = image.resize((224, 224))
        # generate empty space for confidence socre
        add_confi = Image.new('L', (224,224+font_size), 255)
        # edge detect and converting it to white background
        edges = feature.canny(np.array(image)/255.0, sigma=2)
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", font_size, encoding="unic")
        image = Image.fromarray(edges)
        image.mode = 'L'
        image = ImageOps.invert(image)
        add_confi.paste(image, (0,font_size))
        # put confidence score input the image
        draw = ImageDraw.Draw(add_confi)
        (x, y) = (30, 0)
        message = str(round(float(confidence[i-1]),4))
        draw.text((x, y), message, fill=0, font=font)
        x = (i-1)*224
        new_im.paste(add_confi,(x,0))

    new_im.save(result_path+"/"+model+"/"+str(rank+1).zfill(2)+"_"+str(class_name)+".jpg")

def plot_concept_importance(concept_stats, importance_stats, network, specification, compared_layer='11', plotby = 'iou', out_dir='.',
                            alblation_type='zero_out', sortby='value', scale='auto'):
    if alblation_type == 'zero_out':
        Acc_type = 'Accuracy_Drop'
    elif alblation_type == 'keep':
        Acc_type = 'Accuracy'
    layer_detail = pd.read_csv(f'result/tally/{network}/tally{compared_layer}.csv')
    layer_name = layerNames(network)
    # Find intersection of network-wise concept and Conv5 concept
    # due to Conv5 doses contain all concepts in the network, this importance plot can only show that ones that
    # contained by Conv5
    # unique_concept = importance_stats[f'Concept_{network}'].unique()
    inter_concepts = pd.read_csv(
        f'result/tally/inter_concepts/alexnet_alexnet-r_{compared_layer}.csv').Concept.to_list()
    concept_stats = concept_stats[inter_concepts].sort_values(by=f'tally{compared_layer}', axis=1, ascending=False)

    # concept_list = concept_stats.columns.to_list()
    concept_list = inter_concepts
    importance_values = pd.DataFrame(np.zeros((1, len(concept_list))), index=[f'{Acc_type}'], columns=concept_list)
    # collect comparison data
    for concept in concept_list:
        if plotby == 'iou':
            channel_id = layer_detail[layer_detail.label == concept].sort_values(by='score').iloc[-1].unit - 1
            # convert to percentage and round
            importance_values.loc[Acc_type, concept] = round(importance_stats.loc[channel_id, network] * 100, 2)
        elif plotby == 'max':
            importance = importance_stats[importance_stats[f'Concept_{network}'] == concept].sort_values(by=network)[network].iloc[-1]
            importance_values.loc[Acc_type, concept] = round(importance * 100, 2)
        elif plotby == 'mean':
            if len(importance_stats[importance_stats[f'Concept_{network}'] == concept][network]) == 1:
                row = importance_stats[importance_stats[f'Concept_{network}'] == concept]
                importance = (row[network]/row.Count)[0]
                importance_values.loc[Acc_type, concept] = round(importance * 100, 2)
            else:
                importance = importance_stats[importance_stats[f'Concept_{network}'] == concept][network].mean()
                importance_values.loc[Acc_type, concept] = round(importance * 100, 2)

    # ploting
    if sortby == 'values':
        values_concept = concept_stats.loc[f'tally{compared_layer}'].to_numpy(dtype=np.int32)
        values_importance = importance_values.loc[Acc_type].to_numpy(dtype=np.float32)
        xticks = concept_stats.columns.to_list()
    elif sortby == 'concept':
        values_concept = concept_stats.sort_index(axis=1).loc[f'tally{compared_layer}'].to_numpy(dtype=np.int32)
        values_importance = importance_values.sort_index(axis=1).loc[Acc_type].to_numpy(dtype=np.float32)
        xticks = concept_stats.sort_index(axis=1).columns.to_list()

    # plot iconcepts
    mycm = cm.get_cmap('plasma', values_concept.size)
    canvas_length = max((values_concept.size)*0.3, 16)
    plt.rcParams['figure.figsize'] = [canvas_length, 5]
    rect = plt.bar(np.arange(values_concept.size), values_concept, color=mycm.colors)
    plt.autoscale(axis='y')
    plt.xticks(np.arange(values_concept.size), xticks, rotation=45, ha='right')
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(3.0))
    # plt.tick_params(axis='x', labelsize=8)
    plt.title(f'{network} Concept in {layer_name[compared_layer]}')
    autolabel(rect)
    plt.tight_layout()
    img_name_concept = f'{out_dir}/{network}_{specification}_{layer_name[compared_layer]}_concept.jpg'
    plt.savefig(img_name_concept)
    plt.close('all')

    # plot importance
    # mycm = cm.get_cmap('plasma', values_concept.size)
    plt.rcParams['figure.figsize'] = [canvas_length, 5]
    rect = plt.bar(np.arange(values_concept.size), values_importance, color=mycm.colors)
    if scale == 'auto':
        plt.autoscale(axis='y')
    else:
        plt.ylim(scale)
    plt.xticks(np.arange(values_concept.size), xticks, rotation=45, ha='right')
    plt.title(f'{network} {layer_name[compared_layer]} {Acc_type}(%)  Mean: {values_importance.mean():0.2f}  std: {(np.std(values_importance)):0.2f}')
    autolabel(rect, 'float', text_float=0, keep=1)
    plt.tight_layout()
    img_name_acc = f'{out_dir}/{network}_{specification}_{layer_name[compared_layer]}_{Acc_type}.jpg'
    plt.savefig(img_name_acc)
    plt.close('all')

    os.system(f'montage -quiet {img_name_concept} {img_name_acc} -tile 1x2 -geometry +0+0 '
              f'{out_dir}/{network}_{specification}_{Acc_type}_{plotby}.jpg')
    os.system(f'rm {img_name_concept} {img_name_acc}')

    return f'{out_dir}/{network}_{specification}_{Acc_type}_{plotby}.jpg'

''' Scatter plot (for accuracy drop)'''
def scatterPlot(stats, xtick, out_path, plot_name, img_size='auto', ylabel='Accuracy drop in (%)', networks=['AlexNet', 'AlexNet-R'], title=None, pltheight=4, fontsize=14, scale=None):
    x = range(len(xtick))
    mycm = cm.get_cmap('Set3', 2)
    if img_size == 'auto':
        plt.figure(figsize=(len(xtick)*.3, pltheight))
    else:
        plt.figure(figsize=img_size)

    plt.plot(x, [stats[0].mean()]*len(xtick), '--', color=mycm.colors[0], alpha=0.8)
    plt.plot(x, [stats[1].mean()] * len(xtick), '--', color='#ceb301', alpha=0.8)
    plt.scatter(x, stats[0], c=mycm.colors[0].reshape(1, -1), marker='o', s=80, label=f'{networks[0]}', edgecolors='k')
    plt.scatter(x, stats[1], c=mycm.colors[1].reshape(1, -1), marker='^', s=80, label=f'{networks[1]}', edgecolors='k')
    if scale is not None:
        plt.ylim(scale)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(x, xtick, fontsize=fontsize, rotation=45, ha='right')
    plt.legend(loc=0, prop={'size': fontsize})

    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, plot_name))
    plt.close('all')

''' scatterPlot keep first ten channels for all for plot and put them together'''
def scatterPlot2(stats, xtick, out_path, plot_name, img_size='auto', ylabel='Accuracy drop in (%)',
                 networks=['AlexNet', 'AlexNet-R'], title=None, pltheight=4, fontsize=14, topk=10 , scale=None):
    x = range(len(xtick[:topk]))
    mycm = cm.get_cmap('Set3', 2)
    if img_size == 'auto':
        plt.figure(figsize=(len(xtick[:topk])*.3, pltheight))
    else:
        plt.figure(figsize=img_size)

    for idx, stats_pair in enumerate(stats):
        if idx == 0:
            plt.subplot(1,4,1)
            plt.plot(x, [stats_pair[0].mean()] * len(xtick[:topk]), '--', color=mycm.colors[0], alpha=0.8)
            plt.plot(x, [stats_pair[1].mean()] * len(xtick[:topk]), '--', color='#ceb301', alpha=0.8)
            plt.scatter(x, stats_pair[0][:topk], c=mycm.colors[0].reshape(1, -1), marker='o', s=80, label=f'{networks[0]}', edgecolors='k')
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.scatter(x, stats_pair[1][:topk], c=mycm.colors[1].reshape(1, -1), marker='^', s=80, label=f'{networks[1]}', edgecolors='k')
            plt.xticks(x, xtick, fontsize=fontsize, rotation=45, ha='right')
            plt.legend(loc=0, prop={'size': fontsize})
            if title is not None:
                plt.title(title[idx])
            if scale is not None:
                plt.ylim(scale)
        else:
            plt.subplot(1,4,idx+1)
            plt.plot(x, [stats_pair[0].mean()] * len(xtick[:topk]), '--', color=mycm.colors[0], alpha=0.8)
            plt.plot(x, [stats_pair[1].mean()] * len(xtick[:topk]), '--', color='#ceb301', alpha=0.8)
            plt.scatter(x, stats_pair[0][:topk], c=mycm.colors[0].reshape(1, -1), marker='o', s=80,  edgecolors='k')
            plt.scatter(x, stats_pair[1][:topk], c=mycm.colors[1].reshape(1, -1), marker='^', s=80,  edgecolors='k')
            plt.xticks(x, xtick, fontsize=fontsize, rotation=45, ha='right')
            if title is not None:
                plt.title(title[idx])
            if scale is not None:
                plt.ylim(scale)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, plot_name))
    plt.close('all')

''' Scatter plot for pair of data'''
def scatterPlotPair(stats, xtick, out_path, plot_name, img_size='auto', ylabel='Mean TV', networks=['AlexNet', 'AlexNet-R'], title=None, pltheight=4, fontsize=14, scale=None):
    x = range(len(xtick))
    mycm = cm.get_cmap('Set3', 2)
    if img_size == 'auto':
        plt.figure(figsize=(len(xtick)*.3, pltheight))
    else:
        plt.figure(figsize=img_size)
    plt.plot(x, stats[0], '--', c=mycm.colors[0], zorder=1)
    plt.plot(x, stats[1], '--', c=mycm.colors[1], zorder=1)

    plt.scatter(x, stats[0], c=mycm.colors[0].reshape(1, -1), marker='o', s=80, label=f'{networks[0]}', edgecolors='k', zorder=2)
    plt.scatter(x, stats[1], c=mycm.colors[1].reshape(1, -1), marker='^', s=80, label=f'{networks[1]}', edgecolors='k', zorder=2)
    if scale is not None:
        plt.ylim(scale)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(x, xtick, fontsize=fontsize, rotation=45, ha='right')
    plt.legend(loc=0, prop={'size': fontsize})

    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, plot_name))
    plt.close('all')
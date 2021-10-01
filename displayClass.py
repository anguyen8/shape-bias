from plotHelper import display_class_confidence, load_labelMap
import os
import pandas as pd


temp = "result/class"
if not os.path.isdir(temp):
    os.mkdir(temp)

# plot all
cate = pd.read_csv("cate_diff.csv", index_col=0)
models = cate.columns.to_list()
labelMap = load_labelMap()

for model in models:
    print("Saving images for "+model)
    path = temp + "/" + model
    if not os.path.isdir(path):
        os.mkdir(path)
    cate_list = cate[model].to_list()[0][1:-1].split(",")
    for i, className in enumerate(cate_list):
        display_class_confidence(model, int(className), i, labelMap)
        print("Saving: "+str(i+1)+"/20")



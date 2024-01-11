import json
import os
import pandas as pd
def convert(img_size, box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    return (x1, y1, x2, y2)
def decode_json(json_floder_path, json_name,label):
    txt_name = r'这里存放即将生成txt文件的路径/' + json_name[0:-5] + '.txt'
    txt_file = open(txt_name, 'w')
    json_path = os.path.join(json_floder_path, json_name)
    data = json.load(open(json_path, 'r'))
    img_w = data['imageWidth']
    img_h = data['imageHeight']
    for i in data['shapes']:
        if i['shape_type'] == 'rectangle':
            if (label['label'] != i['label']).all():
                new_label=pd.DataFrame(columns=['label'], data=[i['label']])
                label=label.append(new_label,ignore_index=True)
            try:
                x1 = float((i['points'][0][0])) / img_w
                y1 = float((i['points'][0][1])) / img_h
                x2 = float((i['points'][1][0])) / img_w
                y2 = float((i['points'][1][1])) / img_h
                n = label[label['label']==i['label']].index[0]
                bb = (x1, y1, x2, y2)
                bbox = convert((img_w, img_h), bb)
                txt_file.write(str(n) + " " + " ".join([str(a) for a in bbox]) + '\n')
            except IndexError:
                print(json_name[0:-5]+'的'+i['label']+"标签坐标缺失")
    return label
if __name__ == "__main__":
    json_floder_path = r'这里存放你存json文件的路径/'
    json_names = os.listdir(json_floder_path)
    label= pd.DataFrame(columns = ['label'])
    for json_name in json_names:
        if json_name[-4:]=='json':
            print(json_name)
            label=decode_json(json_floder_path, json_name,label)

    label.to_csv('label.txt', sep='\t', index=True)

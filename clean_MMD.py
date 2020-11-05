import os,sys,csv,json

keep=['55122','44111']

thresh=5.000
above=0
total=0

res_csv = sys.argv[1]
gt_json = sys.argv[2]
new_json = sys.argv[3]

with open(gt_json) as f:
    data = json.load(f)
train_data ={d['img_path']:d['gt'] for d in data['train']}

with open(res_csv) as f:
    csvreader = csv.reader(f, delimiter=',', quotechar='"')
    for row in csvreader:
        image, gt, pred, cer = row
        total+=1
        #print('{} -- {}'.format(cer,thresh))
        if float(cer)>thresh:
            print('{}: {}  GT:{}  PRED:{}'.format(image,cer,gt,pred))
            above+=1
            try:
                del train_data[image]
            except KeyError:
                print('unable to remove '+image)

print('removed {}/{} : {}'.format(above,total,float(above)/total))
data['train'] = [{'img_path':path,'gt':gt} for path,gt in train_data.items()]
with open(new_json,'w') as f:
    json.dump(data,f)

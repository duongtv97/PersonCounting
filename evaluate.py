import glob, os

output_name = "txt_output_n100_1"   #Directory of result videos
os.chdir(output_name)

file_list = []
ap = []
for fi in glob.glob("*.txt"):
    file_list.append(fi)
file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

os.chdir("..")

def IOU(i,j,gt_dict,r_dict):
    
    return len(gt_dict[i].intersection(r_dict[j]))/len(gt_dict[i].union(r_dict[j]))
    
    
for name in file_list:
    gt = open('GROUND_TRUTH/'+name,"r")
    
    results = open(output_name+"/"+name,"r")
    indx=0
    gt_id = set()
    r_id = set()
    
    gt_id_list = []
    r_id_list = []
    
    for i,j in zip(gt,results):
        s_gt = i.split()
        s_results = j.split()
        #print (s_gt)
        gt_id.add(s_gt[3])
        r_id.add(s_results[3])
    
    for i,j in zip(gt_id,r_id):
        gt_id_list.append([i])
        r_id_list.append([j])    
    
    #print (gt_id_list)
    gt_id_dict = dict.fromkeys(gt_id)
    r_id_dict = dict.fromkeys(r_id)
    
    
    for i in gt_id_dict:
        gt_id_dict[i] = set()
    
    for j in r_id_dict:    
        r_id_dict[j] = set()
    
    
    for i in gt_id_dict:
        gt = open("GROUND_TRUTH/"+name,"r")
        #results = open("txt_output/result_test.txt","r")
        for sgt in gt:
            sgt = sgt.split()
            #sr = sr.split()
            if (i==sgt[3]):
                gt_id_dict[i].add(sgt[1]+sgt[2]+sgt[0])
                
           
    
    for j in r_id_dict:
        results = open(output_name+"/"+name,"r")
        for sr in results:
            sr = sr.split()              
            if (j==sr[3]):           
                r_id_dict[j].add(sr[1]+sr[2]+sr[0])
    
    
    
        
    correct = 0
    for i in gt_id_dict:
        iou_list = []
        #print ("===========")
        for j in r_id_dict:
            
            iou = IOU(i,j,gt_id_dict,r_id_dict)   
            #print("iou: ", iou)     
            iou_list.append(iou)
            #print ("iou list: ", iou_list)
            #print ("len r_id_dict: ", len(r_id_dict))
        if len(iou_list)==len(r_id_dict) and max(iou_list) >= 0.5 :
            correct += 1
    print ("num of correction: ", correct)
    print ("total id: ", len(gt_id))        
    eval_result = correct/len(gt_id)
    ap.append(eval_result)
    print("Precision of "+ name + " is: ", eval_result)

import numpy as np
aver = np.asarray(ap)
print ("==================")
print ("AP: ", np.average(aver))                              

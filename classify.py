import torch
import numpy as np

label_list = []
gt_label = []

def predict_rule(test_dataset, test_vis, DEVICE, model):
    for i in range(len(test_dataset)):
        image, gt_mask = test_dataset[i]
        image_vis = test_vis[i][0].astype('uint8')
        

        gt_mask = gt_mask.squeeze()
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        area_not_zero = []
        gt_area = []

        for j in range(0,16):
            area = np.sum(pr_mask[j])
            area_not_zero.append(area)
            area2 = np.sum(gt_mask[j])
            gt_area.append(area2)

            
        #percemtages
        # unknown = area_not_zero[0]   
        alkaline_feldspar_gt = gt_area[0]*100/(np.sum(gt_area)+1)
        biotite_gt = gt_area[1]*100/(np.sum(gt_area)+1)
        calcite_gt = gt_area[2]*100/(np.sum(gt_area)+1)
        feldspar_gt = gt_area[3]*100/(np.sum(gt_area)+1)
        hornblende_gt = gt_area[4]*100/(np.sum(gt_area)+1)
        nepheline_gt = gt_area[5]*100/(np.sum(gt_area)+1)
        nosean_gt = gt_area[6]*100/(np.sum(gt_area)+1)
        olivine_gt = gt_area[7]*100/(np.sum(gt_area)+1)
        orthoclase_gt = gt_area[8]*100/(np.sum(gt_area)+1)
        phlogopite_gt = gt_area[9]*100/(np.sum(gt_area)+1)
        plagioclase_gt = gt_area[10]*100/(np.sum(gt_area)+1)
        pyroxene_gt = gt_area[11]*100/(np.sum(gt_area)+1)
        quartz_gt = gt_area[12]*100/(np.sum(gt_area)+1)
        sanidine_gt = gt_area[13]*100/(np.sum(gt_area)+1)
        titanaugite_gt = gt_area[14]*100/(np.sum(gt_area)+1)
        
        
        if phlogopite_gt > 8:
            label = '金伯利岩'
            gt_label.append(label)
        elif olivine_gt > 55:
            label = '橄榄岩'
            gt_label.append(label)
        elif orthoclase_gt > 10:
            label = '正长岩'
            gt_label.append(label)
        elif titanaugite_gt > 95:
            label = '钛辉石岩'
            gt_label.append(label)
        elif calcite_gt > 10:
            label = '方解岩'
            gt_label.append(label)
        elif pyroxene_gt > 46:
            label = '辉石岩'
            gt_label.append(label)
        elif plagioclase_gt > 95:
            label = '斜长岩'
            gt_label.append(label)
        elif plagioclase_gt > 60:
            label = '长岩'
            gt_label.append(label)
        elif plagioclase_gt > 30 and quartz_gt == 0:
            label = '长岩'
            gt_label.append(label)
        elif olivine_gt > 10:
            label = '橄榄岩'
            gt_label.append(label)
        elif feldspar_gt > 90:
            label = '长岩'
            gt_label.append(label)
        elif sanidine_gt > 90:
            label = '粗面岩'
            gt_label.append(label)
        elif alkaline_feldspar_gt > 50 and nepheline_gt > 10:
            label = '霞石岩'
            gt_label.append(label)
        elif quartz_gt > 50:
            label = '石英岩'
            gt_label.append(label)
        elif (nepheline_gt + pyroxene_gt) > 90:
            label = '霞石岩'
            gt_label.append(label)
        elif nepheline_gt > 10:
            label = '霞石岩'
            gt_label.append(label)
        elif nosean_gt > 8:
            label = '黝方石响岩'
            gt_label.append(label)
        elif hornblende_gt > 60:
            label = '角闪岩'
            gt_label.append(label)
        elif biotite_gt > 27:
            label = '云煌岩'
            gt_label.append(label)
        elif (alkaline_feldspar_gt + plagioclase_gt) > 80:
            label = '长岩'
            gt_label.append(label)
        elif (alkaline_feldspar_gt + plagioclase_gt + quartz_gt) > 55:
            label = '花岗岩'
            gt_label.append(label)
        elif alkaline_feldspar_gt > 50:
            label = '伟晶岩'
            gt_label.append(label)
        elif quartz_gt > 23:
            label = '石英岩'
            gt_label.append(label)
        elif plagioclase_gt > 12:
            label = '长岩'
            gt_label.append(label)
        else:
            print(i)
            visualize(
                image = image_vis
            )

        #percemtages
        
        alkaline_feldspar = area_not_zero[0]*100/(np.sum(area_not_zero)+1)
        biotite = area_not_zero[1]*100/(np.sum(area_not_zero)+1)
        calcite = area_not_zero[2]*100/(np.sum(area_not_zero)+1)
        feldspar = area_not_zero[3]*100/(np.sum(area_not_zero)+1)
        hornblende = area_not_zero[4]*100/(np.sum(area_not_zero)+1)
        nepheline = area_not_zero[5]*100/(np.sum(area_not_zero)+1)
        nosean = area_not_zero[6]*100/(np.sum(area_not_zero)+1)
        olivine = area_not_zero[7]*100/(np.sum(area_not_zero)+1)
        orthoclase = area_not_zero[8]*100/(np.sum(area_not_zero)+1)
        phlogopite = area_not_zero[9]*100/(np.sum(area_not_zero)+1)
        plagioclase = area_not_zero[10]*100/(np.sum(area_not_zero)+1)
        pyroxene = area_not_zero[11]*100/(np.sum(area_not_zero)+1)
        quartz = area_not_zero[12]*100/(np.sum(area_not_zero)+1)
        sanidine = area_not_zero[13]*100/(np.sum(area_not_zero)+1)
        titanaugite = area_not_zero[14]*100/(np.sum(area_not_zero)+1)
    
    
        if phlogopite > 8:
            label = '金伯利岩'
            label_list.append(label)
        elif olivine > 55:
            label = '橄榄岩'
            label_list.append(label)
        elif orthoclase > 10:
            label = '正长岩'
            label_list.append(label)
        elif titanaugite > 95:
            label = '钛辉石岩'
            label_list.append(label)
        elif calcite > 10:
            label = '方解岩'
            label_list.append(label)
        elif pyroxene > 46:
            label = '辉石岩'
            label_list.append(label)
        elif plagioclase > 95:
            label = '斜长岩'
            label_list.append(label)
        elif plagioclase > 60:
            label = '长岩'
            label_list.append(label)
        elif plagioclase > 30 and quartz == 0:
            label = '长岩'
            label_list.append(label)
        elif olivine > 10:
            label = '橄榄岩'
            label_list.append(label)
        elif feldspar > 90:  
            label = '长岩'
            label_list.append(label)
        elif sanidine > 90:
            label = '粗面岩'
            label_list.append(label)
        elif alkaline_feldspar > 50 and nepheline > 10:
            label = '霞石岩'
            label_list.append(label)
        elif quartz > 50:
            label = '石英岩'
            label_list.append(label)
        elif (nepheline + pyroxene) > 90:
            label = '霞石岩'
            label_list.append(label)
        elif nepheline > 10:
            label = '霞石岩'
            label_list.append(label)
        elif nosean > 8:
            label = '黝方石响岩'
            label_list.append(label)
        elif hornblende > 60:
            label = '角闪岩'
            label_list.append(label)
        elif biotite > 27:
            label = '云煌岩'
            label_list.append(label)
        elif (alkaline_feldspar + plagioclase) > 80:
            label = '长岩'
            label_list.append(label)
        elif (alkaline_feldspar + plagioclase + quartz) > 55:
            label = '花岗岩'
            label_list.append(label)
        elif alkaline_feldspar > 50:
            label = '伟晶岩'
            label_list.append(label)
        elif quartz > 23:
            label = '石英岩'
            label_list.append(label)
        elif plagioclase > 12:
            label = '长岩'
            label_list.append(label)
        else:
            index = []

            for j in range(len(area_not_zero)):
                if area_not_zero[j] != 0:
                    index.append(j)
                    
            print(index)

            if len(index) > 0:
                values = []
                for k in index:
                    if k == 0:
                        v1 = abs(alkaline_feldspar - 50)
                        values.append(v1)
                    elif k == 1:
                        v1 = abs(biotite - 27)
                        values.append(v1)
                    elif k == 2:
                        v1 = abs(calcite - 10)
                        values.append(v1)
                    elif k == 3:
                        v1 = abs(feldspar - 90)
                        values.append(v1)
                    elif k == 4:
                        v1 = abs(hornblende - 60)
                        values.append(v1)
                    elif k == 5:
                        v1 = abs(nepheline - 10)
                        values.append(v1)
                    elif k == 6:
                        v1 = abs(nosean - 8)
                        values.append(v1)
                    elif k == 7:
                        v1 = abs(olivine - 10)
                        values.append(v1)
                    elif k == 8:
                        v1 = abs(orthoclase - 10)
                        values.append(v1)
                    elif k ==9:
                        v1 = abs(phlogopite - 8)
                        values.append(v1)
                    elif k ==10:
                        v1 = abs(plagioclase - 30)
                        values.append(v1)
                    elif k==11:
                        v1 = abs(pyroxene - 46)
                        values.append(v1)
                    elif k==12:
                        v1 = abs(quartz - 23)
                        values.append(v1)
                    elif k==13:
                        v1 = abs(sanidine - 90)
                        values.append(v1)
                    elif k==14:
                        v1 = abs(titanaugite  - 95)
                        values.append(v1)
                print(values)
                arr_idx = np.array(values).argmin()
                if index[arr_idx] == 0:
                    label = '伟晶岩'
                    label_list.append(label)
                elif index[arr_idx] == 1:
                    label = '云煌岩'
                    label_list.append(label)
                elif index[arr_idx] == 2:
                    label = '方解岩'
                    label_list.append(label)
                elif index[arr_idx] == 3:
                    label = '长岩'
                    label_list.append(label)
                elif index[arr_idx] == 4:
                    label = '角闪岩'
                    label_list.append(label)
                elif index[arr_idx] == 5:
                    label = '霞石岩'
                    label_list.append(label)
                elif index[arr_idx] == 6:
                    label = '黝方石响岩'
                    label_list.append(label)
                elif index[arr_idx] == 7:
                    label = '橄榄岩'
                    label_list.append(label)
                elif index[arr_idx] == 8:
                    label = '正长岩'
                    label_list.append(label)
                elif index[arr_idx] ==9:
                    label = '金伯利岩'
                    label_list .append(label)
                elif index[arr_idx] ==10:
                    label = '斜长岩'
                    label_list.append(label)
                elif index[arr_idx]==11:
                    label = '辉石岩'
                    label_list.append(label)
                elif index[arr_idx]==12:
                    label = '石英岩'
                    label_list.append(label)
                elif index[arr_idx]==13:
                    label = '粗面岩'
                    label_list.append(label)
                elif index[arr_idx]==14:
                    label = '钛辉石岩'
                    label_list.append(label)          
            else:
                print("else")
                lablelist = ['伟晶岩', '花岗岩', '二长岩', '云煌岩', '角闪岩', '金伯利岩', '橄榄岩', '正长岩', '钛辉石岩', '方解岩', '辉石岩', '斜长岩', '长岩', '粗面岩', '霞石岩', '石英岩', '黝方石响岩']
                n = np.random.choice(len(lablelist))
                label = lablelist[n]
                label_list.append(label)
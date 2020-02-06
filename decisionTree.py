from __future__ import division
import csv
import sys
import math
import numpy as np

from matplotlib import pyplot as plt

# with open('education_train.tsv') as tsvfile:
#      reader = csv.reader(tsvfile, delimiter ='\t')
#      List = list(reader)


def gini(label1,label2, total):
 if total==0:
     return 0
 else:

    if label2 == 0 or label1 ==0:
      gini_final = 0
    else:
        gini_final =  2*((label1)/(total))*((label2)/(total))
    return gini_final


def getCount(dataset):

    array_List =np.asarray(dataset)
    columns = len(array_List[0])

    yes, no, yeslabel1, yeslabel2, nolabel1, nolabel2 = [], [], [], [], [], []
    for j in range(columns):
        yes_condition = array_List[1, j]
        classes = np.asarray(array_List[1:len(array_List), -1])
        class_1 = classes[1]
        index_rows_yes = (np.argwhere(array_List[1:, j] == yes_condition))
        yes.append(len(index_rows_yes))
        count_of_yes_classes1 = np.sum(classes[index_rows_yes] == class_1)
        yeslabel1.append(count_of_yes_classes1)
        count_of_yes_classes2 = len(index_rows_yes) - count_of_yes_classes1
        yeslabel2.append(count_of_yes_classes2)
        index_rows_no = np.argwhere(array_List[1:, j] != array_List[1, j])
        no.append(len(index_rows_no))
        count_of_no_classes1 = np.sum(classes[index_rows_no] != classes[1])
        count_of_no_classes2 = len(index_rows_no) - (count_of_no_classes1)
        nolabel1.append(count_of_no_classes1)
        nolabel2.append(count_of_no_classes2)

    return yes, yeslabel1, yeslabel2, no, nolabel2, nolabel1

def majority_vote(data):

    rows = len(data)
    columns = len(data[0])
    label1, label2 = 0,0
    for i in range(1, rows):
         if data[1][columns-1] == data[i][columns-1]:
                label1 += 1
         else:
               label2_name = data[i][columns-1]
               label2 +=1
    if label1>label2:
        return data[1][columns-1], label1, label2
    elif label1 == label2:
        alphabets = [data[1][columns - 1], label2_name]
        new = sorted(alphabets)
        #print new[1]
        return new[1], label1, label2
    else:
        return label2_name, label1, label2

def split_data(index, dataset):  #split data into left and right


    left, right = [], []

    for i in range(1, len(dataset)):
        if dataset[i][index] == dataset[1][index]:
            left.append(list(dataset[i]))
        else: right.append(list(dataset[i]))
    left.insert(0, list(dataset[0]))
    right.insert(0, list(dataset[0]))
    return left, right
    #print left


def gini_gain(dataset, max_depth):

 columns = len(dataset[0])
 rows = len(dataset)
 pno= [0] * (columns)
 pyes = [0] * (columns)
 gini_gain= [0] * (columns)
 gini_yes = [0]*(columns)
 gini_no = [0]*columns
 left_data = [0]*columns
 right_data = [0]*columns

 yes, yeslabel1, yeslabel2, no, nolabel1, nolabel2 = getCount(dataset)
 gini_bigdata = gini(yes[columns - 1], no[columns - 1], (rows - 1))
 for j in range(0, columns-1):
      node_rows= [], []
      gini_yes[j] = gini(yeslabel1[j], yeslabel2[j], yes[j])
      gini_no[j] = gini(nolabel1[j], nolabel2[j], no[j])
      gini_gain[j]= gini_bigdata - (((float(no[j])/ float(rows-1))* gini_no[j]) + ((float(yes[j])/float(rows-1)) * gini_yes[j]))


 max_index = gini_gain.index(max(gini_gain))
 #print("Index:" + str(max_index))
 if max(gini_gain)>0:
            #print gini_gain


            left_data, right_data = split_data(max_index, dataset)
     #print majority_vote(left_data), majority_vote(right_data)
     #print(left_data)   
            label_left, y1lcount, y2lcount = majority_vote(left_data)
            label_right, y1rcount, y2rcount = majority_vote(right_data)
            attribute_left  = left_data[1][max_index]
            attribute_right = right_data[1][max_index]

            if max_depth <= 0 or (y1lcount == 0 or y2lcount == 0) or (y1rcount == 0 or y2rcount == 0):
                terminal = True
            else:
                terminal = False

            return {'best_index': max_index, 'attribute_name' :dataset[0][max_index], 'left': left_data,
         'right': right_data, 'decleft': label_left, 'decright': label_right, "not_negative": True,
          'attribute_left': attribute_left, 'attribute_right': attribute_right, "y1lcount": y1lcount, "y2lcount": y2lcount, "y1rcount":y1rcount, "y2rcount":y2rcount,
         'terminal': terminal}

 else:

        return {'not_negative': False}
#info_gain(List)
    #return max(set(max_labels), key=max_labels.count) if max_labels else 0

def create_splits(node, max_depth):
    #iterate through tree to implement majority vote/ terminal node
 #print("the current depth is " + str(max_depth))
 if (node['not_negative']):
  if ((max_depth - 1) >= 0):
     #left, right = node['left'], node['right']
     left = node['left']
     right = node['right']
     decleft, decright = node['decleft'], node['decright']
 
     #keep deleting the created nodes (drop previous node)
     #del(node['right'], node['left'])


     if (node['y1lcount'] == 0 or node['y2lcount'] == 0):
         pass
         # print(node['attribute_name'])
         # print('left attribute is:' + node['attribute_left'] + ' ' + 'and' + ' ' + 'decision is:'  + '' + decleft + ' y1count= ' + str(node['y1lcount']) + ' y2count= ' + str(node['y2lcount']))
         # #node['left'] = majority_vote(left)
         #print(node['left'])
         #print("classified")
     else:
         # print(node['attribute_name'])
         # print('left attribute is:' + node['attribute_left'] + ' ' + 'and' + ' ' + 'decision is:'  + '' + decleft + ' y1count= ' + str(node['y1lcount']) + ' y2count= ' + str(node['y2lcount']))
         # print("left_split")
         node['left'] = gini_gain(left, max_depth-1)
         create_splits(node['left'], (max_depth - 1))

 #if len(right) <= min_record:
     if (node['y1rcount'] == 0 or node['y2rcount'] == 0):
         pass
         # print(node['attribute_name'])
         # print('right attribute is:' + node['attribute_right']  + ' ' + 'and' + ' ' + 'decision is:' + ''  + decright + ' y1count= ' + str(node['y1rcount']) + ' y2count= ' + str(node['y2rcount']))
         # print("classified")
     else:
         # print(node['attribute_name'])
         # print('right attribute is:' + node['attribute_right']  + ' ' + 'and' + ' ' + 'decision is:' + ''  + decright + ' y1count= ' + str(node['y1rcount']) + ' y2count= ' + str(node['y2rcount']))
         # print("right_split")
         node['right'] = gini_gain(right, max_depth-1)
         #print(node['right'])
         create_splits(node['right'], (max_depth-1))

def create_tree(dataset, max_depth):
    columns = len(dataset[0])
    if max_depth> columns - 1:
        max_depth = columns -1
    root = gini_gain(dataset, max_depth)
    #print(root)
    create_splits(root, max_depth)
    return root


#tree = create_tree(List, 3)
#print_tree(tree, 0)
def predict(root, row, dataset, dataset_chota, max_depth, currentdepth):
    #print(root['terminal'])
    #print("salim" + str(row))
  if root:

        if max_depth == 0:
            result1, count1, count2 = majority_vote(dataset_chota)
            return result1

            #print (root['best_index'])
            #print("length" + str(len(dataset)))
            #left1, right1 = split_data(root['best_index'], dataset_chota)
            #print root['attribute_left']
            #index = root['best_index']
        elif   max_depth != 0 :
            if currentdepth==max_depth:
                #print root['best_index']
                left1, right1 = split_data(root['best_index'], dataset_chota)
                index = root['best_index']

                #print ('yehuaa')
                #print("sdsaddasd" + str(row) +str(index))
                #print(len(dataset))
                #print(str(row) + ":row_val and " + str(index) + str(root['attribute_left']))
                if dataset[row][index] == root['attribute_left']:
                        result1, result2, result3 = majority_vote(left1)
                    #print("jumanji" + str(result1))
                        return result1
                else:
                        result1, result2, result3 = majority_vote(right1)
                    #print("tarzan" + str(result1))
                        return result1
            elif  currentdepth <max_depth:

                    left1, right1 = split_data(root['best_index'], dataset_chota)
                    index = root['best_index']
                    if  dataset[row][index] == root['attribute_left'] and (root['y1lcount'] ==0 or root['y2lcount'] == 0):

                        result1, result2, result3 = majority_vote(left1)
                        #print ('yehua')
                        return result1

                    elif dataset[row][index] == root['attribute_right'] and (root['y1rcount'] ==0 or root['y2rcount'] == 0):


                        result1, result2, result3 = majority_vote(right1)
                    # print ('yehua')
                        return result1


                    else:
                            if dataset[row][index] == root['attribute_left']:
                                save_list, count1, count2 = majority_vote(left1)

                                #left1, right1 = split_data(root['best_index'], dataset_chota)
                                base = root['left']
                                #print ('base1')
                                if base['not_negative'] == False:
                                    #print("gulabo")
                                    return save_list
                                else:
                                    #print ('ok1')
                                    result1 = predict(root['left'], row, dataset, left1, max_depth, currentdepth + 1)

                                    return result1


                    #print("hulalala23")
                    #print(root['left'])
                    # print("hulLllas")
                    # root_left = info_gain(root['left'], max_depth, current_depth + 1)

                            else:
                                     save_list2, count1, count2 = majority_vote(right1)
                                     #left1, right1 = split_data(root['best_index'], dataset_chota)
                                     base = root['right']
                                     #print (base)

                                     if base['not_negative'] == False:

                                        #result1, result2, result3 = majority_vote(left1)
                                         #print("dssdsds")
                                         return save_list2
                                     else:   #print ('ok')
                                         result1 = predict(root['right'], row, dataset, right1, max_depth, currentdepth + 1)
                                     return result1

                    #print("sitar")

                    #print(root['right'])
                    # root_right = info_gain(root['right'], max_depth, current_depth + 1)

            else: print ('current>max')
  else: print('kuchbhi')
# predict_results = []
# #print predict(tree, 1, List, List, 4, 1)
# for i in range(1, len(List)):
#     # for i in range(24,25):
def ma():

    # train_input = sys.argv[1]
    # test_input = sys.argv[2]
    # maxdepth = sys.argv[3]
    # train_output = sys.argv[4]
    # test_output = sys.argv[5]
    # metric = sys.argv[6]
    train_input = "politicians_train.tsv"
    test_input = "politicians_test.tsv"
    #maxdepth = 2

    train_output = "pol_train.txt"
    test_output = "pol_test.txt"
    metric = "metric.txt"
    with open(train_input) as tsvfile:
        train_ini= list(csv.reader(tsvfile, delimiter = '\t'))
    with open(test_input) as tsvfile:
        test_ini= list(csv.reader(tsvfile, delimiter = '\t'))
    max_depthlist =[]
    error_trainplot, error_testplot = [], []
    for m in range(len(train_ini[0])-1):
        max_depthlist.append(m)
        max_depth = m
        # print max_depth
    #trainout = open(train_output, 'w+')
    #testout = open(test_output, 'w+')
    #metric = open(metric, 'w+')
        predict_resultstrain,predict_results,  error_trainlist, testoutlist, error_testlist, non_errortrain, non_errortest, trainoutlist = [], [], 0, [], 0, 0, 0, []
        tree = create_tree(train_ini, max_depth)

        for i in range(1, len(train_ini)):
        # for i in range(24,25):
            predict_resultstrain.append(predict(tree, i, train_ini, train_ini, max_depth, 1))

        for i in range(0, len(predict_resultstrain)):
            if predict_resultstrain[i] != train_ini[i + 1][len(train_ini[0]) - 1]:
                error_trainlist += 1

            else:
                non_errortrain += 1
        # for i in range(len(predict_resultstrain)):
        #     trainout.write(predict_resultstrain[i] + '\n')

        for i in range(1, len(test_ini)):

            predict_results.append(predict(tree, i, test_ini, train_ini, max_depth, 1)) #check
        # for i in range(len(predict_results)):
        #     testout.write(predict_results[i] + '\n')

    # print (predict_results)
        for i in range(0, len(predict_results)):
            if predict_results[i] != test_ini[i + 1][len(test_ini[0]) - 1]:
                error_testlist += 1

            else:
                non_errortest += 1

        error_train = (error_trainlist / (len(train_ini) - 1))
        error_test = (error_testlist/ (len(test_ini)-1))

        error_trainplot.append(error_train)
        error_testplot.append(error_test)
    print error_testplot, error_trainplot

    plt.plot(max_depthlist, error_trainplot, label = 'Training error')
    plt.plot(max_depthlist, error_testplot, label = 'Test error')
    plt.legend(loc = 'upper right')
    plt.xlabel('max_depth')
    plt.ylabel('error')
    plt.show()
    #metric.write("error(train): " + str(error_train) + '\n' + 'error(test): ' + str(error_test))

if __name__=='__main__':
    ma()




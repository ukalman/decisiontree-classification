# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 16:23:57 2021

@author: Umut
"""
from pprint import pprint
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
import sklearn.metrics as metrics

df = pd.read_csv("diabetes_data_upload.csv")
#pd.set_option("display.max_rows", None, "display.max_columns", None)
new_df = df.groupby("Gender")
for key, item in new_df:
    print(new_df. get_group(key))

columns = [column for column in df.columns]

arr = df.to_numpy()

kf = KFold(n_splits=5, shuffle=True)

X_train = np.zeros([416, 16])
X_test = np.zeros([104, 16])
y_train = np.zeros([416, 1])
y_test = np.zeros([104, 1])

def discretization(arr):

    for row_index in range(arr.shape[0]):
        row_value = arr[row_index][0]
                
        if(row_value >= 16 and row_value < 54):
            arr[row_index][0] = "1"
        
        elif(row_value >= 54 and row_value < 91):
            arr[row_index][0] = "2"
            
        
    

   
discretization(arr)

print(arr)



def entropy(array):

    total = array[0] + array[1]
    pos_pro = array[0] / total
    neg_pro = array[1] / total

    if(pos_pro == 0):
        return 0
    if(neg_pro == 0):
        return 0
    first = (-1 * pos_pro) * math.log(pos_pro,2)
    second = (-1 * neg_pro) * math.log(neg_pro,2)
    return first + second

def entropy_total(y_train):
    pos_count = 0
    neg_count = 0
    for i in range(1,len(y_train)):
        if (y_train[i] == "Positive"):
            pos_count+=1
        
        elif (y_train[i] == "Negative"):
            neg_count+=1
        
    return entropy([pos_count,neg_count])

def pos_neg_counter(col_num,attr_value,X_train,y_train):

    pos_count = 0
    neg_count = 0

    for i in range(1,len(X_train)):
        if(X_train[i][col_num] == attr_value):
            if(y_train[i] == "Positive"):
                pos_count+=1
            
            elif(y_train[i] == "Negative"):
                neg_count+=1
    
    return [pos_count,neg_count]

def information_gain(col_num,X_train,y_train):
      
    collection_entropy = entropy_total(y_train)
    attribute_values = [] 
    
    
    for i in range(1,len(X_train)):
        try:
            
            if(X_train[i][col_num] not in attribute_values):
                attribute_values.append(X_train[i][col_num])
        
        except:
            print("HATA")
            print("X_train[i][col_num]",X_train[i][col_num])
            print("attribute_values: ",attribute_values)
            quit()

    subtractive = 0
    for value in attribute_values:
        pos_neg_arr = pos_neg_counter(col_num,value,X_train,y_train)
        attr_amount = pos_neg_arr[0] + pos_neg_arr[1]
        proportion = attr_amount / (len(y_train) -1)
        factor = entropy(pos_neg_arr)
        multi = proportion * factor
        subtractive += multi

    return (collection_entropy - subtractive)


def modify_arr(X_train,y_train,attr_name,attr_value):
    new_X_array = [row[:] for row in X_train]
    new_y_array = y_train[:]
    removal_rows = []
    attr_index = 0
    
    for i in range(X_train.shape[1]):
        if X_train[0][i] == attr_name:
            attr_index = i
            break
    
    for j in range(1,len(X_train)):
        if X_train[j][attr_index] != attr_value:
            removal_rows.append(X_train[j])
            
    for row in removal_rows:

        for j in range(1,len(new_X_array)):
            #print("bu row: ",row)
            #print("bu da new_X_array[j]: ",new_X_array[j])
            if (new_X_array[j] == row).all():
                
                new_X_array = np.delete(new_X_array,j,0)
                new_y_array = np.delete(new_y_array,j)
                break
        
    
    # Remove that column
    new_X_array = np.delete(new_X_array,attr_index,1)
    
    return [new_X_array,new_y_array]

def get_index(X_train,attr_name):
    for i in range(X_train.shape[1]):
        if X_train[0][i] == attr_name:
            return i

def classification(tree_branches,X_test,y_test):
    classifications = []

    for i in range(1,X_test.shape[0]):
        deger = search(tree_branches,tree_branches[0],X_test,i)
        classifications.append(deger)
        
    
    for i in range(len(classifications)):
        if classifications[i] not in ["Positive","Negative"]:
            print("EVET MAALESEF NONE TYPE VAR.")
            classifications[i] = "Negative"
        
    return classifications
            
            
def search(tree_branches,branch,X_test,i):
    
    primary_key = list(branch.keys())[0]
    index = get_index(X_test,primary_key)
    value = X_test[i][index]
    for key in list(branch[primary_key]):
        if key == value:
            if branch[primary_key][key] == "Positive":
                return "Positive"
                
            elif branch[primary_key][key] == "Negative":
                return "Negative"
                
            
            branch_index = 0
 
            for j in range(len(tree_branches)):
                if tree_branches[j] == branch:
                     branch_index = j
                     break

            for nearest_index in range(branch_index,len(tree_branches)):
                if list(tree_branches[nearest_index].keys())[0] == branch[primary_key][key]:
                    return search(tree_branches,tree_branches[nearest_index],X_test,i)
                    
        

class DecisionTree():

    def __init__(self,X_train,y_train):
        self.tree = None
        self.root = None
        self.tree_size = 0
        self.X_train = X_train
        self.y_train = y_train
        self.branches = list()
        self.used_attributes = []
            
    def size(self):
        return len(self.branches)
    
    def build_tree_ID3(self,attr_name = None):
        
        
        end_value = True
        for branch in self.branches:
            if(type(list(branch.values())[0]) != type(dict())):
                    end_value = False
                    break
                
        if(end_value == True and len(self.branches) != 0):
            return
        
        
        if(self.size() == 0):
            self.tree = dict()
            best_attr,attr_index = self.find_best_attribute(self.X_train,self.y_train)
            self.root = best_attr
            self.tree_size += 2
            
            attr_values = []
            for col_value in self.X_train[1:,attr_index]:
                if col_value not in attr_values:
                    attr_values.append(col_value)
                    
            self.tree[best_attr] = dict()
            
            self.branches.append(dict())
            self.branches[0][best_attr] = dict()

            temp_X_array = np.array([row[:] for row in self.X_train])
            unmodified_X_array = np.array([row[:] for row in temp_X_array])

            for value in attr_values:
                if(self.check_attribute2(temp_X_array, y_train, attr_index, value)):

                    best_node,node_index = self.find_best_attribute(modify_arr(temp_X_array,y_train,best_attr,value)[0],modify_arr(temp_X_array, y_train, best_attr, value)[1])
                    
                    self.branches[0][best_attr][value] = best_node
                    self.branches.append({best_node: list()})
                    
                    self.branches[len(self.branches)-1][best_node].append(modify_arr(unmodified_X_array,y_train,best_attr,value)[0])
                    self.branches[len(self.branches)-1][best_node].append(modify_arr(unmodified_X_array,y_train,best_attr,value)[1])
                    
                    index = self.get_index(temp_X_array,best_node)
                    
                    temp_X_array = np.delete(temp_X_array,index,1)
                else:
                    for i in range(1,len(temp_X_array)):
                        if temp_X_array[i][attr_index] == value:
                            self.branches[0][best_attr][value] = y_train[i]
                            break
                    
                    
            self.build_tree_ID3()
        
        else:
            temp_X_array = np.array([row[:] for row in self.X_train])
            
            unBranched = 0
            for i in range(self.size()):
                if type(list(self.branches[i].values())[0]) == type(list()):
                    unBranched = i
                    attr_name = list(self.branches[i])[0]
                    temp_X_array = list(self.branches[i].values())[0][0]
                    temp_y_array = list(self.branches[i].values())[0][1]
                    break

            attr_values = []
            attr_index = self.get_index(temp_X_array, attr_name)
            for col_value in temp_X_array[1:,attr_index]:
                if col_value not in attr_values:
                    attr_values.append(col_value)
            

            unmodified_X_array = np.array([row[:] for row in temp_X_array])
            unmodified_y_array = np.array([row for row in temp_y_array])
            
       
            
            self.branches[unBranched][attr_name] = dict()
            for value in attr_values:
                
                
                attr_index = self.get_index(temp_X_array, attr_name)
                if(self.check_attribute2(temp_X_array, temp_y_array, attr_index, value)):
                    best_node,node_index = self.find_best_attribute(modify_arr(temp_X_array,temp_y_array,attr_name,value)[0],modify_arr(temp_X_array, temp_y_array, attr_name, value)[1])
                
                    
                    self.branches[unBranched][attr_name][value] = best_node
                    self.branches.append({best_node: list()})
                    
                
                    self.branches[len(self.branches)-1][best_node].append(modify_arr(unmodified_X_array,unmodified_y_array,attr_name,value)[0])
                    self.branches[len(self.branches)-1][best_node].append(modify_arr(unmodified_X_array, unmodified_y_array, attr_name, value)[1])

                    index = self.get_index(temp_X_array,best_node)

                    try:
                        temp_X_array = np.delete(temp_X_array,index,1)
                    except:
                        print("Attr name: ",attr_name)
                        print("temp_X_array: ",temp_X_array)
                        print("temp_y_array: ",temp_y_array)
                        print("index: ",index)
                        quit()
                    
                else:

                    self.branches[unBranched][attr_name][value] = modify_arr(unmodified_X_array, unmodified_y_array, attr_name, value)[1][1]
 
            self.build_tree_ID3()
                    

    def get_index(self,X_train,attr_name):
        for i in range(X_train.shape[1]):
            if X_train[0][i] == attr_name:
                return i
    
    def find_best_attribute(self,X_train,y_train):

        attr_gain = [0,0,0]
        for i in range(X_train.shape[1]):
            gain = information_gain(i,X_train,y_train)

            if gain > attr_gain[2]:
                attr_gain[2] = gain
                attr_gain[0] = X_train[0][i]
                attr_gain[1] = i
        
        if(attr_gain[0] == 0):
            attr_gain[0] = X_train[0][0]
            attr_gain[1] = 0

        return attr_gain[0],attr_gain[1]
    
    
    def check_attribute2(self,X_train,y_train,attr_index,attr_value):
        pos_neg_arr = pos_neg_counter(attr_index,attr_value,X_train,y_train)
        entropy_value = entropy(pos_neg_arr)
        if (entropy_value <= 0 or X_train.shape[1] == 1):
            return False
        return True
        
    def print_again(self):
        print(self.tree)
    
    def print_tree(self,root,nodes):
        print("root: ",root)
        return_str = "" + root
        if self.size() > 1:
            return_str += " --> ["
            for i in range(len(nodes)):
                if isinstance(nodes[i],list):
                            continue
                
                if i != len(nodes) -1: 
                
                    if not isinstance(nodes[i+1],list):
                        return_str += nodes[i] + " "
                    else:
                        return_str += self.print_tree(nodes[i],nodes[i+1])
                else:
                    return_str += nodes[i]
            return_str += "] "
        return return_str



for train, test in kf.split(arr):

    X_train = arr[train, :-1]
    y_train = arr[train, -1]
    X_test = arr[test, :-1]
    y_test = arr[test, -1]
    
    X_train = np.insert(X_train,0,columns[:16],axis = 0)
    y_train = np.insert(y_train,0,columns[16],axis = 0)
    
    X_test = np.insert(X_test,0,columns[:16],axis = 0)
    y_test = np.insert(y_test,0,columns[16],axis = 0)
         
    tree = DecisionTree(X_train,y_train)
    tree.build_tree_ID3()
    
    
    
    classify_results = classification(tree.branches, X_test, y_test)
    
    
    acc_score = metrics.accuracy_score(y_test[1:],classify_results)
    print("ACCURACY: ",acc_score)
    
        
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:00:12 2018

@author: chenj
"""
import pandas as pd
from pandas.core.frame import DataFrame
import math
import re
import sys

def readArff(fileName):  
    arffFile = open(fileName,'r')  
    data = []; d={}  
    for line in arffFile.readlines():  
        if not (line.startswith('@')):  
            if not (line.startswith('%')):  
                if line !='\n':  
                    L=line.strip('\n')  
                    k=L.split(',')  
                    data.append(k)                      
        else:
            if (line.startswith("@attribute")):
                J=line.strip('\n')
                J=re.sub('[^a-zA-Z0-9+-]',' ', J)
                J=re.sub('\s+',' ',J)
                J=J.split(" ")[:-1]
                d[J[2]]=J[3:]
    arffFile.close()
    data=DataFrame(data);feature=list(d.keys())
    data.columns=feature
    for index in feature:
        if d[index]==[]:
            data[index]=pd.to_numeric(data[index])
    return [data,feature,d]

def InfoGain_nomial(data,feature):
    """
    to calculate the nomial feature Information Gain
    """
    labels=readArff("credit_train.arff")[2]
    data_p=data.loc[data["class"]=="+"]
    data_n=data.loc[data["class"]=="-"]
    E=-(data_p.shape[0]/data.shape[0])*math.log2(data_p.shape[0]/data.shape[0])-(data_n.shape[0]/data.shape[0])*math.log2(data_n.shape[0]/data.shape[0])
    sum=0;labels=labels[feature]
    for i in labels:
        temp=data.loc[data[feature]==i]
        if temp.shape[0]==0:
            H=0
        else:#if temp.shape[0]>=m:#there are fewer than m training instances reaching the node, where m is provided as input to the program,
            if len(temp["class"].unique())==1:##all of the training instances reaching the node belong to the same class
                temp_1=temp.loc[temp["class"]==temp["class"].iloc[0]]
                H=-(temp_1.shape[0]/temp.shape[0])*math.log2(temp_1.shape[0]/temp.shape[0])
                H=(temp.shape[0]/data.shape[0])*H
            else:
                temp_p=temp.loc[temp["class"]=="+"]
                temp_n=temp.loc[temp["class"]=="-"]
                H=-(temp_p.shape[0]/temp.shape[0])*math.log2(temp_p.shape[0]/temp.shape[0])-(temp_n.shape[0]/temp.shape[0])*math.log2(temp_n.shape[0]/temp.shape[0])
                H=(temp.shape[0]/data.shape[0])*H
        sum=H+sum
    return [E-sum,labels]

def InfoGain_numeric(data,feature):
    """
    to calculate the numeric feature Information Gain
    """
    if len(data[feature].unique())==1:
        return [data[feature].iloc[0],float("-inf")]
    data_p=data.loc[data["class"]=="+"]
    data_n=data.loc[data["class"]=="-"]
    E=-(data_p.shape[0]/data.shape[0])*math.log2(data_p.shape[0]/data.shape[0])-(data_n.shape[0]/data.shape[0])*math.log2(data_n.shape[0]/data.shape[0])
    index=data[feature].tolist()
    index=sorted(index)
    points=[];answer=float("-inf")
    for i in range(len(index)-1):
        points.append((index[i]+index[i+1])/2)
    count=0
    for i in range(len(index)):
        if index[i]==max(index):
           count+=1       
    for i in range(len(points)-count+1):#if temp.shape[0]>=m:#there are fewer than m training instances reaching the node, where m is provided as input to the program,
        temp=data.loc[data[feature]>points[i]] 
        if len(temp["class"].unique())==1:#all of the training instances reaching the node belong to the same class
            temp_1=temp.loc[temp["class"]==temp["class"].iloc[0]]
            H_p=-(temp_1.shape[0]/temp.shape[0])*math.log2(temp_1.shape[0]/temp.shape[0])
            H_p=(temp.shape[0]/data.shape[0])*H_p
        else:
            temp_p=temp.loc[temp["class"]=="+"]
            temp_n=temp.loc[temp["class"]=="-"];
            H_p=-(temp_p.shape[0]/temp.shape[0])*math.log2(temp_p.shape[0]/temp.shape[0])-(temp_n.shape[0]/temp.shape[0])*math.log2(temp_n.shape[0]/temp.shape[0])
            H_p=(temp.shape[0]/data.shape[0])*H_p
        temp=data.loc[data[feature]<=points[i]]
        if len(temp["class"].unique())==1:##all of the training instances reaching the node belong to the same class
            temp_1=temp.loc[temp["class"]==temp["class"].iloc[0]]
            H_n=-(temp_1.shape[0]/temp.shape[0])*math.log2(temp_1.shape[0]/temp.shape[0])
            H_n=(temp.shape[0]/data.shape[0])*H_n
        else:
            temp_p=temp.loc[temp["class"]=="+"]
            temp_n=temp.loc[temp["class"]=="-"]
            H_n=-(temp_p.shape[0]/temp.shape[0])*math.log2(temp_p.shape[0]/temp.shape[0])-(temp_n.shape[0]/temp.shape[0])*math.log2(temp_n.shape[0]/temp.shape[0])
            H_n=(temp.shape[0]/data.shape[0])*H_n
        sum=H_n+H_p
        if (E-sum)>answer:
            answer=E-sum;
            best_p=points[i]
    return [best_p,answer]

def FindBestSplit(train_data,all_feature):
    answer=float("-inf");list_node=[]
    for i in all_feature:
        if train_data[i].dtypes=="O":
            IG=InfoGain_nomial(train_data,i)[0]
        else:
            IG=InfoGain_numeric(train_data,i)[1]
        if IG>answer:
            best_node=i
            answer=IG
    zero=False      
    if answer<0:
        zero=True#no feature has positive information gain
    else:
        if train_data[best_node].dtypes=="O":
            list_node.append(best_node);list_node.append(InfoGain_nomial(train_data,best_node)[1]);list_node.append(zero)
        else:
            list_node.append(best_node);list_node.append(InfoGain_numeric(train_data,best_node)[0]);list_node.append(zero)
    return list_node

def DetermineCandidateSplits(train_data,best_node):
    if best_node!=None:
        index=train_data.columns.tolist()[:-1]
        index.remove(best_node)
    else:
        index=train_data.columns.tolist()[:-1]
    return index

def MakeSubetree(train_data,m,zero=False,tree_model=[],d={},best_node=None):
    if train_data.shape[0]<m or len(train_data["class"].unique())==1 or zero:
        if train_data.shape[0]==0:
            key=list(d.keys())[0]
            for index in tree_model:
                if key in list(index.keys())[1:]:
                    target=index
                    for i in range(len(list(index.keys()))):
                        if key==list(index.keys())[i]:
                            order=list(index.keys())[i-1];
            np=target[order][1]
            nn=target[order][2]
            if np<nn:
                d["sign"]="-"
            else:
                d["sign"]="+"
            tree_model.append(d)
            d={}
            a=[tree_model,d]
            return a
        elif len(train_data["class"].unique())==1:
            sign=train_data["class"].iloc[0]
            d["sign"]=sign
            tree_model.append(d)
            d={}
            a=[tree_model,d]
            return a
        else:
            num_p=train_data.loc[train_data["class"]=="+"].shape[0];
            num_n=train_data.loc[train_data["class"]=="-"].shape[0];
            if num_p>=num_n:
                d["sign"]="+"
                tree_model.append(d)
                d={}
                a=[tree_model,d]
                return a
            else:
                d["sign"]="-"
                tree_model.append(d)
                d={}
                a=[tree_model,d]
                return a       
    else:
        all_feature=DetermineCandidateSplits(train_data,best_node)
        ans=FindBestSplit(train_data,all_feature)
        zero=ans[-1];best_node=ans[0]
        if train_data[best_node].dtypes=="O":
            labels=ans[1];
            C={}
            for i in labels:
                c=train_data.loc[train_data[best_node]==i]
                C[i]=c
            for l in labels:
                train_data=C[l]
                num_p=train_data.loc[train_data["class"]=="+"].shape[0];
                num_n=train_data.loc[train_data["class"]=="-"].shape[0];
                d[best_node]=[l,num_p,num_n,"=="+" "+"'"+l+"'"]
                all_feature=DetermineCandidateSplits(train_data,best_node)
                d=MakeSubetree(train_data,m,zero,tree_model,d,best_node)[-1]
        else:
            p=ans[1]
            D_s=train_data.loc[train_data[best_node]<=p]
            D_b=train_data.loc[train_data[best_node]>p]
            D={"<=":D_s,">":D_b}
            for ope in D.keys():
                train_data=D[ope]
                num_p=train_data.loc[train_data["class"]=="+"].shape[0];
                num_n=train_data.loc[train_data["class"]=="-"].shape[0];
                tag=best_node+" "+str(p)
                d[tag]=[p,num_p,num_n,ope+' '+str(p)]
                d=MakeSubetree(train_data,m,zero,tree_model,d)[-1]
    d={}       
    return [tree_model,d]

def print_the_tree(model):
   """
   the function to print the tree out
   """
   for i in range(len(model[0].keys())-1):
       key=list(model[0].keys())[i]
       model[0][key].append(i)
   for i in range(1,len(model)):
       sample_all=model[:i]
       k=list(model[i].keys())[0]
       for index in sample_all:
           if k in list(index.keys()):
               num=index[k][-1];
               for j in range(len(model[i])-1):
                   model[i][list(model[i].keys())[j]].append(num+j) 
   for index in model:
       for i in range(len(index)): 
           if i < len(index)-2:
               num=index[list(index.keys())[i]][-1]
               interval=str(index[list(index.keys())[i]][1:3])
               interval=re.sub('[,]','', interval)
               pr_k=str(list(index.keys())[i].split()[0])
               dirt=index[list(index.keys())[i]][3]
               dirt=re.sub('[\']','', dirt)
               if "==" in dirt:
                   dirt=dirt[1:]
               print(num*"|\t"+pr_k.lower()+" "+dirt+" "+interval)
           elif i == len(index)-2:
               num=index[list(index.keys())[i]][-1]
               interval=str(index[list(index.keys())[i]][1:3])
               interval=re.sub('[,]','', interval)
               pr_k=str(list(index.keys())[i].split()[0])
               dirt=index[list(index.keys())[i]][3]
               dirt=re.sub('[\']','', dirt)
               if "==" in dirt:
                   dirt=dirt[1:]
               print(num*"|\t"+pr_k.split()[0].lower()+" "+dirt+" "+interval+' '+str(index[list(index.keys())[i+1]]))    
   return model

def dictionary_slicing(dictionary,end):
    s={}
    for key in list(dictionary.keys())[:end]:
        s[key]=dictionary[key]
    return s

def transfor_1(model):
   final=[]
   final.append(model[0])
   for i in range(1,len(model)):
       sample_all=final[:i]
       k=list(model[i].keys())[0]
       for index in sample_all:
           if k in list(index.keys()):
               temp={}
               for j in range(len(index)):
                   if k == list(index.keys())[j]:
                       nu=j;tar=index
                       break
               break
       temp=dictionary_slicing(tar,nu)
       temp.update(model[i])
       final.append(temp)
   return final

def transfor_2(model):
    new=[]
    for index in model:
        temp=""
        for i in range(len(index)):
            if i <len(index)-2:
                key=list(index.keys())[i]
                cont=repr(key.split()[0])
                temp=temp+"test["+cont+"].iloc[i]"+" "+str(index[key][3])+" and "
            elif i == len(index)-2:
                key=list(index.keys())[i]
                cont=repr(key.split()[0])
                temp=temp+"test["+cont+"].iloc[i]"+" "+str(index[key][3])
            else:
                key=list(index.keys())[i]
                temp=temp+index[key]
        new.append(temp)
    return new   

def predict(model,test):
    actual=test["class"]
    del test["class"]
    print("<Predictions for the Test Set Instances>")
    predicted=[];j=0
    for i in range(test.shape[0]):
        for index in model:
            if eval(index[:-1]):
                predicted.append(index[-1])
                break
        print(str(i+1)+": Actual: "+actual.iloc[i]+" Predicted: "+str(predicted[i]))
        if actual.iloc[i]==predicted[i]:
            j=j+1
    print("Number of correctly classified: "+str(j)+" Total number of test instances: "+str(test.shape[0]))
    return predicted

def final_predit(train_data,test_data,m):
    use=readArff(train_data)
    tree=use[0]#get the traindata
    use_test=readArff(test_data)
    tree_test=use_test[0]#get the test data
    tree_model_final=MakeSubetree(tree,m,tree_model=[],d={})[0]
    mid=print_the_tree(tree_model_final)
    final=transfor_1(mid)
    real_final=transfor_2(final)
    result=predict(real_final,tree_test)
    return result
    
#final_predit("credit_train.arff","credit_test.arff",30)

def main():
    data_train_filename = sys.argv[1]
    data_test_filename = sys.argv[2]
    m = int(sys.argv[3])
    final_predit(data_train_filename , data_test_filename , m)
    
if __name__ == '__main__':
    main()       








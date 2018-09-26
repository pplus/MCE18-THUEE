#*- coding:UTF-8 -*-
"""
##  ==========================================================================
##
##       author : Liang He, heliang@mail.tsinghua.edu.cn
##                Xianhong Chen, chenxianhong@mail.tsinghua.edu.cn
##   descrption : sre14 preprocess for SO/MO sGPLDA
##                This script is based on
##                NIST SRE14 offical released script (cosine scoring).
##      created : 20180923
## last revised : 
##
##    Liang He, +86-13426228839, heliang@mail.tsinghua.edu.cn
##    Aurora Lab, Department of Electronic Engineering, Tsinghua University
##  ==========================================================================
"""

import numpy as np
import scipy.io
from sklearn.metrics import roc_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import LPLDA

def load_ivector(filename):
    utt = np.loadtxt(filename,dtype='string',delimiter=',',skiprows=1,usecols=[0])
    ivector = np.loadtxt(filename,dtype='float32',delimiter=',',skiprows=1,usecols=range(1,601))
    spk_id = []
    for iter in range(len(utt)):
        spk_id = np.append(spk_id,utt[iter].split('_')[0])

    return spk_id, utt, ivector

def length_norm(mat):
# length normalization (l2 norm)
# input: mat = [utterances X vector dimension] ex) (float) 8631 X 600

    norm_mat = []
    for line in mat:
        temp = line/np.math.sqrt(sum(np.power(line,2)))
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)
    return norm_mat

def make_spkvec(mat, spk_label):
# calculating speaker mean vector
# input: mat = [utterances X vector dimension] ex) (float) 8631 X 600
#        spk_label = string vector ex) ['abce','cdgd']

#     for iter in range(len(spk_label)):
#         spk_label[iter] = spk_label[iter].split('_')[0]

    spk_label, spk_index  = np.unique(spk_label,return_inverse=True)
    spk_mean=[]
    mat = np.array(mat)

    # calculating speaker mean i-vector
    for i, spk in enumerate(spk_label):
        spk_mean.append(np.mean(mat[np.nonzero(spk_index==i)],axis=0))
    spk_mean = length_norm(spk_mean)
    return spk_mean, spk_label

def calculate_EER(trials, scores):
# calculating EER of Top-S detector
# input: trials = boolean(or int) vector, 1: postive(blacklist) 0: negative(background)
#        scores = float vector

    # Calculating EER
    fpr,tpr,threshold = roc_curve(trials,scores,pos_label=1)
    fnr = 1-tpr
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # print EER_threshold
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr+EER_fnr)
    
    print "Top S detector EER is %0.2f%%"% (EER*100)
    return EER

def get_trials_label_with_confusion(identified_label, groundtruth_label,dict4spk,is_trial ):
# determine if the test utterance would make confusion error
# input: identified_label = string vector, identified result of test utterance among multi-target from the detection system 
#        groundtruth_label = string vector, ground truth speaker labels of test utterances
#        dict4spk = dictionary, convert label to target set, ex) train2dev convert train id to dev id

    trials = np.zeros(len(identified_label))
    for iter in range(0,len(groundtruth_label)):
        enroll = identified_label[iter].split('_')[0]
        test = groundtruth_label[iter].split('_')[0]
        if is_trial[iter]:
            if enroll == dict4spk[test]:
                trials[iter]=1 # for Target trial (blacklist speaker)
            else:
                trials[iter]=-1 # for Target trial (backlist speaker), but fail on blacklist classifier
                
        else :
            trials[iter]=0 # for non-target (non-blacklist speaker)
    return trials

def find_neighbors_same_count(vecs_mean, vecs, vecs_not):
    """find nearest neighbors, same count

    Parameters
    ----------
    filename : 
        mean of vectors: target class, within class mean
        vectors: within class vectors
        between vectors: between class vectors

    Returns
    -------
    nearest neighbor vectors
    """    
    
    w_count = len(vecs)
    vecs_not_sim = [np.dot(vecs_mean, vecs_not[i]) for i in range(0,len(vecs_not))]
    vecs_not_label = np.argsort(vecs_not_sim)
    vecs_not_label = vecs_not_label[::-1]
    neighbors_vecs = [vecs_not[vecs_not_label[i]] for i in range(0,w_count)]
    
    return np.array(neighbors_vecs)

def compute_neighbor(vectors, labels):
    """compute between vectors

    Parameters
    ----------
    filename : 
        development and train vectors
        labels

    Returns
    -------
    between vectors
    """    

    if len(vectors) != len(labels):
        print ('len(vectors) != len(labels)')
        exit(-1)

    unique_labels = np.unique(labels)
    print (len(labels), len(unique_labels))        
        
    b_vectors = []
    b_labels = []
    
    for label in unique_labels:
        
        vecs = [vectors[i] for i in range(len(vectors)) if labels[i] == label]
        vecs_not = [vectors[i] for i in range(len(vectors)) if labels[i] != label]
        
        ## nearest selection
        vecs_mean = np.mean(vecs, axis=0)
        vecs_neighbors = find_neighbors_same_count(vecs_mean, vecs, vecs_not)
        
#        ## random selection
#        vecs_neighbors = find_random_select(vecs_not, len(vecs))
                
        if len(b_vectors) == 0:
            b_vectors = np.vstack((vecs, vecs_neighbors))
            b_labels = label*np.ones((len(vecs) + len(vecs_neighbors),1))
        else:
            b_vectors = np.vstack((b_vectors, np.vstack((vecs, vecs_neighbors))))
            b_labels = np.vstack((b_labels, label * np.ones((len(vecs) + len(vecs_neighbors),1))))
    
        print (len(vecs), len(vecs_neighbors), len(b_vectors), len(vecs_not))
        
    return b_vectors, b_labels


def label_str_to_int(label_str):
    """label, string to int

    Parameters
    ----------
    filename : string label

    Returns
    -------
    int label
    """
    
    label_dict = {}
    label_int = []
    for item in label_str:
        if item not in label_dict.keys():
            label_dict[item] = len(label_dict) + 1
        label_int.append(label_dict[item])
    
    return np.array(label_int)

def calculate_EER_with_confusion(scores,trials):
# calculating EER of Top-1 detector
# input: trials = boolean(or int) vector, 1: postive(blacklist) 0: negative(background) -1: confusion(blacklist)
#        scores = float vector

    # exclude confusion error (trials==-1)
    scores_wo_confusion = scores[np.nonzero(trials!=-1)[0]]
    trials_wo_confusion = trials[np.nonzero(trials!=-1)[0]]

    # dev_trials contain labels of target. (target=1, non-target=0)
    fpr,tpr,threshold = roc_curve(trials_wo_confusion,scores_wo_confusion,pos_label=1, drop_intermediate=False)
    fnr = 1-tpr
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # EER withouth confusion error
    EER = fpr[np.argmin(np.absolute((fnr-fpr)))]
    
    # Add confusion error to false negative rate(Miss rate)
    total_negative = len(np.nonzero(np.array(trials_wo_confusion)==0)[0])
    total_positive = len(np.nonzero(np.array(trials_wo_confusion)==1)[0])
    fp= fpr*np.float(total_negative)  
    fn= fnr*np.float(total_positive) 
    fn += len(np.nonzero(trials==-1)[0])
    total_positive += len(np.nonzero(trials==-1)[0])
    fpr= fp/total_negative
    fnr= fn/total_positive

    # EER with confusion Error
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr+EER_fnr)
    
    print "Top 1 detector EER is %0.2f%% (Total confusion error is %d)"% ((EER*100), len(np.nonzero(trials==-1)[0]))
    return EER

## making dictionary to find blacklist pair between train and test dataset
bl_match = np.loadtxt('../data/bl_matching_dev.csv',dtype='string')
dev2train={}
dev2id={}
train2dev={}
train2id={}

for iter, line in enumerate(bl_match):
    line_s = line.split(',')
    dev2train[line_s[1].split('_')[-1]]= line_s[2].split('_')[-1]
    dev2id[line_s[1].split('_')[-1]]= line_s[0].split('_')[-1]
    train2dev[line_s[2].split('_')[-1]]= line_s[1].split('_')[-1]
    train2id[line_s[2].split('_')[-1]]= line_s[0].split('_')[-1]
    
# Loading i-vector
trn_bl_id, trn_bl_utt, trn_bl_ivector = load_ivector('../data/trn_blacklist.csv')
trn_bg_id, trn_bg_utt, trn_bg_ivector = load_ivector('../data/trn_background.csv')
dev_bl_id, dev_bl_utt, dev_bl_ivector = load_ivector('../data/dev_blacklist.csv')
dev_bg_id, dev_bg_utt, dev_bg_ivector = load_ivector('../data/dev_background.csv')
test_id,   test_utt,   test_ivector   = load_ivector('../data/tst_evaluation.csv')

# Calculating speaker mean vector
spk_mean, spk_mean_label = make_spkvec(trn_bl_ivector, trn_bl_id)

#length normalization
trn_bl_ivector = length_norm(trn_bl_ivector)
trn_bg_ivector = length_norm(trn_bg_ivector)
dev_bl_ivector = length_norm(dev_bl_ivector)
dev_bg_ivector = length_norm(dev_bg_ivector)
test_ivector   = length_norm(test_ivector)

print 'Dev set score using train set :'

# making trials of Dev set
dev_ivector = np.append(dev_bl_ivector, dev_bg_ivector,axis=0)
dev_trials = np.append( np.ones([len(dev_bl_id), 1]), np.zeros([len(dev_bg_id), 1]))
dev_id  = np.append(dev_bl_id, dev_bg_id,axis=0)

## lda
lda_ivector = np.vstack((np.asarray(trn_bl_ivector),np.asarray(trn_bg_ivector)))
lda_label = np.concatenate((np.asarray(trn_bl_id),np.asarray(trn_bg_id)),axis=0)
lda = LinearDiscriminantAnalysis(n_components=500)
# lda = LPLDA.LocalPairwiseLinearDiscriminantAnalysis(n_components=400)
lda.fit(lda_ivector, lda_label)

lda_ivector = lda.transform(lda_ivector)
spk_mean = lda.transform(np.asarray(spk_mean))
dev_ivector = lda.transform(np.asarray(dev_ivector))
test_ivector = lda.transform(np.asarray(test_ivector))
trn_bl_ivector = lda.transform(np.asarray(trn_bl_ivector))

## length norm again
spk_mean = length_norm(spk_mean)
dev_ivector = length_norm(dev_ivector)
test_ivector = length_norm(test_ivector)
trn_bl_ivector = length_norm(trn_bl_ivector)

## read matlab score
matlab_data = scipy.io.loadmat('../temp/mce18_result.mat') 

# Cosine distance scoring
scores = matlab_data['score']
test_scores = spk_mean.dot(test_ivector.transpose())

# Multi-target normalization
blscores = spk_mean.dot(trn_bl_ivector.transpose())
mnorm_mu = np.mean(blscores,axis=1)
mnorm_std = np.std(blscores,axis=1)

for iter in range(np.shape(scores)[1]):
    scores[:,iter]= (scores[:,iter] - mnorm_mu) / mnorm_std
dev_scores = np.max(scores,axis=0)

for iter in range(np.shape(test_scores)[1]):
    test_scores[:,iter]= (test_scores[:,iter] - mnorm_mu) / mnorm_std
test_scores_normed = np.max(test_scores,axis=0)

# Top-S detector EER
dev_EER = calculate_EER(dev_trials, dev_scores)

# divide trial label into target and non-target, plus confusion error(blacklist, fail at blacklist detector)
dev_identified_label = spk_mean_label[np.argmax(scores,axis=0)]
dev_trials_label = np.append( dev_bl_id,dev_bg_id)
dev_trials_utt_label = np.append( dev_bl_utt,dev_bg_utt)

# Top-1 detector EER
dev_trials_confusion = get_trials_label_with_confusion(dev_identified_label, dev_trials_label, dev2train, dev_trials )
dev_EER_confusion = calculate_EER_with_confusion(dev_scores,dev_trials_confusion)

# submission file on Dev set
filename = 'THUEE_dev_fixed_primary.csv'
# filename = 'teamname_fixed_contrastive1.csv'
with open(filename, "w") as text_file:
    for iter,score in enumerate(dev_scores):
        id_in_trainset = dev_identified_label[iter].split('_')[0]
        input_file = dev_trials_utt_label[iter]
        text_file.write('%s,%s,%s\n' % (input_file,score,train2id[id_in_trainset]))

# submission file on Eval set
test_identified_label = spk_mean_label[np.argmax(test_scores,axis=0)]
test_trials_label = test_id
test_trials_utt_label = test_utt

filename = 'THUEE_fixed_primary.csv'
with open(filename, "w") as text_file:
    for iter,score in enumerate(test_scores_normed):
        id_in_trainset = test_identified_label[iter].split('_')[0]
        input_file = test_trials_utt_label[iter]
        text_file.write('%s,%s,%s\n' % (input_file,score,train2id[id_in_trainset]))
        
        
#import csv
#import numpy as np
#import random
#from sklearn import discriminant_analysis
#import scipy.io
#
#def load_model_key(filename, model_ids):
#    """Loads a model key to match order of model_ids
#    from
#    Parameters
#    ----------
#    filename : stringb
#        Path to target_speaker_models.csv
#    model_ids : list
#        List of model ivectorids from model_ivectors.csv
#
#    Returns
#    -------
#    y : array, shaped('n_model_ivecs',)
#        Array with each entry the target_speaker_modelid, in
#        same order as model_ids
#    """
#    
#    # load a reverse lookup from ivectorid->target speaker model
#    id_speaker_lookup = {}
#    for row in csv.DictReader(open(filename, 'r')):
#        for e in row['model_ivectorids[5]'].split():
#            id_speaker_lookup[e] = row['target_speaker_modelid']
#
#    # convert to a simple key array for model ivectors
#    y = []
#    for model_id in model_ids:
#        y.append( id_speaker_lookup[ model_id ] )
#
#    return np.array(y)
#	
#	
#def load_dev_key(dev_lable, dev_ids):
#    """Loads development data label to match order of dev_ivectors' ivector ids
#
#    Parameters
#    ----------
#    dev_lable : string
#        Path to development_data_labels.csv
#     dev_ids : list
#        List of dev_ivectors from dev_ivectors.csv
#
#    Returns
#    -------
#    y : array, shaped('n_dev_ivectors',)
#        Array with each entry the development data speaker_id, in
#        same order as dev_ids
#    """
#    
#    # load a reverse lookup from ivectorid->target speaker model
#    id_speaker_lookup = {}
#    id_speaker_dict = {}
#
#    for row in csv.DictReader(open(dev_lable, 'r')):
#        id_speaker_lookup[row['ivectorid']] = row['speaker_id']
#
#        if row['speaker_id'] in id_speaker_dict:
#            id_speaker_dict[row['speaker_id']].append(row['ivectorid'])
#        else:
#            id_speaker_dict[row['speaker_id']] = []
#            id_speaker_dict[row['speaker_id']].append(row['ivectorid'])
#
#    # convert to a simple key array for dev ivectors
#    y = []
#    for dev_id in dev_ids:
#        y.append( id_speaker_lookup[ dev_id ] )
#
#    return [np.array(y), id_speaker_lookup, id_speaker_dict]
#
#
#def load_ivectors(filename):
#    """Loads ivectors
#
#    Parameters
#    ----------
#    filename : string
#        Path to ivector files (e.g. dev_ivectors.csv)
#
#    Returns
#    -------
#    ids : list
#        List of ivectorids
#    durations : array, shaped('n_ivectors')
#        Array of durations for each ivectorid
#    ivectors : array, shaped('n_ivectors', 600)
#        Array of ivectors for each ivectorid
#    """
#    
#    ids = []
#    durations = []
#    ivectors = []
#
#    for row in csv.DictReader(open(filename, 'r')):
#        ids.append( row['ivectorid'] )
#        durations.append( float(row['duration_secs']) )
#        ivectors.append( np.fromstring(row['values[600]'], count=600, sep=' ', dtype=np.float32) )
#
#    return ids, np.array(durations, dtype=np.float32), np.vstack( ivectors )
#    
#    
#def load_test_key(trial_key_file):
#    """Loads test data keys
#
#    Parameters
#    ----------
#    trial_key_file : string
#        Path to trial_key files (e.g. ivec14_sre_trial_key_release.tsv)
#
#    Returns
#    -------
#    key : array
#        a vector of 1(target) or 0(nontarget)
#
#    """
#
#    file_tsv = open(trial_key_file)
#    key = []
#    mask = []
#
#    for line in file_tsv.readlines():
#        if line.split('\t')[2] == 'target':
#            key.append(1)
#        else:
#            key.append(0)
#            
#        if line.strip().split('\t')[3] == 'prog':
#            mask.append(1)
#        elif line.strip().split('\t')[3] == 'eval':
#            mask.append(2)
#        else:
#            mask.append(3)
#
#    file_tsv.close()
#    
#    # the first line is not label, remove it
#    del key[0]
#    del mask[0]
#
#    #convert list  np.array, a vector
#    key = np.array(key).reshape(len(key),1)
#    mask = np.array(mask).reshape(len(mask),1)
#    
#    return [key, mask]
#
#
#def filter_dev_key(dev_ids, id_speaker_lookup, id_speaker_dict, dev_durations):
#    """filter development ivectors
#
#    Parameters
#    ----------
#    filename : dev_ids, id_speaker_lookup, id_speaker_lookup, dev_durations
#
#    Returns
#    -------
#    filtered ivectors
#    """
#
#    if len(dev_ids) != len(dev_durations):
#        exit(-1)
#    
#    y = []
#
#    dev_durations_list = dev_durations
#    dev_durations_list.tolist()
#    
#    dur_reverse_dict = {}
#    for idx in range(len(dev_ids)):
#        dev_id = dev_ids[idx]
#        dur_reverse_dict[dev_id] = dev_durations[idx]
#    
#    for idx in range(len(dev_ids)):
#        
#        dev_id = dev_ids[idx]
#        key_id = id_speaker_lookup[dev_id]
#        seg_count = len(id_speaker_dict[key_id])
#        if seg_count < 3:
#            y.append(0)
#        else:
#            acc_count = 0
#            for item in id_speaker_dict[key_id]:
#                dur = dur_reverse_dict[item]        
#                if dur > 30:
#                    acc_count = acc_count + 1
#            
#            if acc_count > 3:
#                y.append(1)
#            else:
#                y.append(0)
#                
#    return np.array(y)
#
#
#def label_str_to_int(label_str):
#    """label, string to int
#
#    Parameters
#    ----------
#    filename : string label
#
#    Returns
#    -------
#    int label
#    """
#    
#    label_dict = {}
#    label_int = []
#    for item in label_str:
#        if item not in label_dict.keys():
#            label_dict[item] = len(label_dict) + 1
#        label_int.append(label_dict[item])
#    
#    return np.array(label_int)
#
#
#def find_neighbors_same_count(vecs_mean, vecs, vecs_not):
#    """find nearest neighbors, same count
#
#    Parameters
#    ----------
#    filename : 
#        mean of vectors: target class, within class mean
#        vectors: within class vectors
#        between vectors: between class vectors
#
#    Returns
#    -------
#    nearest neighbor vectors
#    """    
#    
#    w_count = len(vecs)
#    vecs_not_sim = [np.dot(vecs_mean, vecs_not[i]) for i in range(0,len(vecs_not))]
#    vecs_not_label = np.argsort(vecs_not_sim)
#    vecs_not_label = vecs_not_label[::-1]
#    neighbors_vecs = [vecs_not[vecs_not_label[i]] for i in range(0,w_count)]
#    
#    return np.array(neighbors_vecs)
#
#def find_random_select(vecs_not, num):
#    """find readom vectors
#
#    Parameters
#    ----------
#    filename : 
#        between vectors
#        number
#
#    Returns
#    -------
#    random selection
#    """    
#
#    random_index = random.sample(range(0,len(vecs_not)), num)
#    random_select = [vecs_not[random_index[i]] for i in range(0,num)]
#    return np.array(random_select)
#
#def compute_neighbor(vectors, labels):
#    """compute between vectors
#
#    Parameters
#    ----------
#    filename : 
#        development and train vectors
#        labels
#
#    Returns
#    -------
#    between vectors
#    """    
#
#    if len(vectors) != len(labels):
#        print ('len(vectors) != len(labels)')
#        exit(-1)
#
#    unique_labels = np.unique(labels)
#    print (len(labels), len(unique_labels))        
#        
#    b_vectors = []
#    b_labels = []
#    
#    for label in unique_labels:
#        
#        vecs = [vectors[i] for i in range(len(vectors)) if labels[i] == label]
#        vecs_not = [vectors[i] for i in range(len(vectors)) if labels[i] != label]
#        
#        ## nearest selection
#        vecs_mean = np.mean(vecs, axis=0)
#        vecs_neighbors = find_neighbors_same_count(vecs_mean, vecs, vecs_not)
#        
##        ## random selection
##        vecs_neighbors = find_random_select(vecs_not, len(vecs))
#                
#        if len(b_vectors) == 0:
#            b_vectors = np.vstack((vecs, vecs_neighbors))
#            b_labels = label*np.ones((len(vecs) + len(vecs_neighbors),1))
#        else:
#            b_vectors = np.vstack((b_vectors, np.vstack((vecs, vecs_neighbors))))
#            b_labels = np.vstack((b_labels, label * np.ones((len(vecs) + len(vecs_neighbors),1))))
#    
#        print (len(vecs), len(vecs_neighbors), len(b_vectors), len(vecs_not))
#        
#    return b_vectors, b_labels
#
#def main():
#
#    print("begin run\n")
#
#    raw_data_path = '../data/'
#    lda_dim = 250
#    
#    #load ivector ids, durations and ivectors (as row vectors)
#    dev_ids, dev_durations, dev_ivec = load_ivectors(raw_data_path+'dev_ivectors.csv')
#    model_ids, model_durations, model_ivec = load_ivectors(raw_data_path+'model_ivectors.csv')
#    test_ids, test_durations, test_ivec = load_ivectors(raw_data_path+'test_ivectors.csv')
#    
#    # load model key corresponding to the same ordering as model_ids
#    model_key = load_model_key(raw_data_path+'target_speaker_models.csv', model_ids)
#    
#    # load development data key corresponding to the same ordering as dev_ids
#    [dev_key, dev_id_speaker_lookup, dev_id_speaker_dict] = load_dev_key(raw_data_path+'development_data_labels.csv', dev_ids)
#    dev_mask = filter_dev_key(dev_ids, dev_id_speaker_lookup, dev_id_speaker_dict, dev_durations)
#        
#    # load development data key corresponding to the same ordering as dev_ids
#    [test_key, test_mask] = load_test_key(raw_data_path+'ivec14_sre_trial_key_release.tsv')
#    
#    # compute the mean and whitening transformation over dev set only
#    m = np.mean(dev_ivec, axis=0)
#    S = np.cov(dev_ivec, rowvar=0)
#    D, V = np.linalg.eig(S)
#    # W = (1/np.sqrt(D) * V).transpose().astype('float32')
#
##    # center and whiten all i-vectors
##    dev_ivec = np.dot(dev_ivec - m, W.transpose())
##    model_ivec = np.dot(model_ivec - m, W.transpose())
##    test_ivec = np.dot(test_ivec - m, W.transpose())
#    
#    # center and whiten all i-vectors
#    dev_ivec = dev_ivec - m
#    model_ivec = model_ivec - m
#    test_ivec = test_ivec - m
#    
#    # dev filter
#    dev_ivec = [dev_ivec[i] for i in range(0,len(dev_ivec)) if dev_mask[i] == 1]
#    dev_key = [dev_key[i] for i in range(0,len(dev_key)) if dev_mask[i] == 1]
#    
#   
#    # train LDA using model_ivec and dev_ivec
#    model_dev_ivec = np.vstack((model_ivec, dev_ivec))
#    model_dev_key = np.hstack((model_key, dev_key))
##    model_dev_ivec = np.vstack(dev_ivec)
##    model_dev_key = np.hstack(dev_key)
#    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=lda_dim)
#    lda.fit(model_dev_ivec, model_dev_key)
#
#    # use lda to convert ivector to low dimension ivector
#    dev_ivec= lda.transform(dev_ivec)
#    model_ivec = lda.transform(model_ivec)
#    test_ivec = lda.transform(test_ivec)
#    
#    avg_model_ivec = np.zeros((len(np.unique(model_key)), model_ivec.shape[1]))
#    avg_model_names = []
#    for i, key in enumerate(np.unique(model_key)):
#        avg_model_ivec[i] = np.mean(model_ivec[model_key == key], axis=0)
#        avg_model_names.append(key)
#
#    # project the converted develepment i-vectors into unit sphere
#    dev_ivec /= np.sqrt(np.sum(dev_ivec ** 2, axis=1))[:, np.newaxis]    
#    model_ivec /= np.sqrt(np.sum(model_ivec ** 2, axis=1))[:, np.newaxis]
#    test_ivec /= np.sqrt(np.sum(test_ivec ** 2, axis=1))[:, np.newaxis]
#    avg_model_ivec /= np.sqrt(np.sum(avg_model_ivec ** 2, axis=1))[:, np.newaxis]
#    
#    dev_ivec = np.vstack((model_ivec, dev_ivec))
#    dev_key = np.hstack((model_key, dev_key))
#    dev_label = label_str_to_int(dev_key)
#    
#    dev_ivec_neighbor, dev_ivec_label = compute_neighbor(dev_ivec, dev_label)
#
#    # save
#    scipy.io.savemat('../temp/sre14.mat', 
#                     mdict={'dev_ivec':dev_ivec, 
#                            'dev_label':dev_label, 
#                            'dev_ivec_neighbor':dev_ivec_neighbor, 
#                            'dev_label_neighbor':dev_ivec_label, 
#                            'enrol_ivec':avg_model_ivec,
#                            'test_ivec':test_ivec,
#                            'test_key':test_key,
#                            'test_mask':test_mask})
#
#if __name__=='__main__':
#    main()
#

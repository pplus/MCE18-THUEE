
%  ==========================================================================
%
%       author : Liang He, heliang@mail.tsinghua.edu.cn
%                Xianhong Chen, chenxianhong@mail.tsinghua.edu.cn
%   descrption : multiobjective optimization training 
%                simplified Gaussian probabilistic 
%                linear discriminant analysis (MOT, sGPLDA)
%                
%      created : 20180206
% last revised : 20180511
%
%    Liang He, +86-13426228839, heliang@mail.tsinghua.edu.cn
%    Aurora Lab, Department of Electronic Engineering, Tsinghua University
%  ==========================================================================

clc;
clear all;
close all;

load '../temp/mce18.mat';

% for wb_fac = 1.1:0.1:2

% for nphi = 50:50:200
    
nphi = 500; % dimension
niter = 50; % iteration
wb_fac = 5.0; % factor

%% train plda
plda = moplda_em_update_sigma(wb_fac, dev_ivec', dev_label, dev_ivec_neighbor', dev_label_neighbor, nphi, niter);

%% score plda
score = score_moplda_trials(plda, enrol_ivec', test_ivec');
save('../temp/mce18_result.mat','score');

% end
% end
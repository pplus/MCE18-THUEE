# MCE18-THUEE


THUEE system for The 1st Multi-target speaker detection and identification Challenge Evaluation \
http://www.mce2018.org/ 


1. prepare data, make directory ./data and ./temp \
put MCE18 offical uncompressed data on "./data/", \
there are "bl_matching_dev.csv, trn_blacklist.csv, trn_background.csv, dev_blacklist.csv, dev_background.csv, tst_evaluation.csv" \
PS: open the website (http://www.mce2018.org/) for data requirement.

There are two systems: \
A. length-normalization + LPLDA + Cosine + score-normalization \
B. lenght-normalization + LDA + PLDA (MotPLDA) + score-normalization


System A: \
A. run ./python/mce18_lplda.py  \
If we use LDA (line 158, "lda = LinearDiscriminantAnalysis(n_components=500)"), the results will be \
Dev set score using train set : \
Top S detector EER is 2.76% \
Top 1 detector EER is 9.20% (Total confusion error is 325)

If we use LPLDA (line 159, "lda = LPLDA.LocalPairwiseLinearDiscriminantAnalysis(n_components=500)"), the results will be \
Dev set score using train set : \
Top S detector EER is 2.41% \
Top 1 detector EER is 8.10% (Total confusion error is 291)


System B: \
A. run ./python/mce18_plda_preprocess.py \
It will generate "./temp/mce18.mat"

Option 1:
B1. run ./matlab/gplda_demo.m for PLDA \
The script will read "./temp/mce18.mat", and it will generate "./temp/mce18_result.mat"

C1. run ./python/mce18_plda_eval.py \
The script will read "./temp/mce18.mat", and the results are \
Dev set score using train set : \
Top S detector EER is 2.77% \
Top 1 detector EER is 9.04% (Total confusion error is 322)


Option 2:
B2. run ./matlab/moplda_demo.m for MoPLDA \
The script will read "./temp/mce18.mat", and it will also generate "./temp/mce18_result.mat"

C2. run ./python/mce18_plda_eval.py \
The script will read "./temp/mce18.mat", and the results are \
Dev set score using train set : \
Top S detector EER is 3.83% \
Top 1 detector EER is 7.50% (Total confusion error is 255)



[1] Suwon Shon, Najim Dehak, Douglas Reynolds, and James Glass, “Mce 2018: The 1st multi-target speaker detection and identification challenge evaluation (mce) plan, dataset and baseline system,” in ArXiv e-prints arXiv:1807.06663, 2018.

[2] L. He, X. Chen, C. Xu, J. Liu, and M. T. Johnson, “Local pairwise linear discriminant analysis for speaker verification,” IEEE Signal Processing Letters, 2018.

[3] L. He, X. Chen, C. Xu, and J. Liu, “Multiobjective Optimization Training of PLDA for Speaker Verification,”
ArXiv e-prints arXiv:1808.08344, Aug. 2018.

He Liang, heliang@mail.tsinghua.edu.cn
Sep. 26, 2018


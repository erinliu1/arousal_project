all fmri data (44 rois)
simple 1-layer GRU with 32 hidden units
uniform segment-level weighting during training
random segment-level 10-fold cross-validation but excluding training segments within X buffer TR of test segments within the same run
 - buffer of 0 TRs match attempt3 segment-level 10-fold CV  
 
fixed seed 42
all fmri data (44 rois)
simple 1-layer GRU with 32 hidden units
uniform segment-level weighting during training
random segment 10-fold cross validation
- segment-level StratifiedKFold with 10 shuffled folds
fixed seed 42
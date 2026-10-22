Exploratory work for fMRI classifier
- used all fmri data (44 rois)
- simple 1-layer GRU with 32 hidden units
- explored different sample weighting during training, including uniform segment-level weighting, participant-balanced weighting, and participant-balanced class-balanced weighting (opt to use participant-balanced and class-balanced weighting in the future)
- fixed seed 42
- explored segment-level CV and the effect of buffering test segments from training segments (opt for LOPO or LORO CV in the future)
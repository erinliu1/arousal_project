After determining participant-balanced class-balanced sampling and LOPO + LORO CV from exploration:
- removed the precentral and postcentral cortical roi's because those are motion-sensitive, leaving a total of 42 ROIs
- simple 1-layer GRU with 32 hidden units
- fix seed 42
- also run correlational analyses for LORO to investigate problems with responsive segment classification
import os
import re
import shutil 

from all_rois import all_rois

def get_participants(base_dir):
    participants = []
    for folder in os.listdir(base_dir):
        if re.match(r'^AAN_\d{2}', folder):
            participants.append(folder)
    return sorted(participants)

def get_runs(base_dir, participant):
    runs = []
    for file in os.listdir(os.path.join(base_dir, participant, 'stcfsl_mc2', 'mclogs')):
        if file.endswith('.mcdat'):
            runs.append(re.match(r'^(run.*?)_', file).group(1))        
    return sorted(runs)

def get_buttonpresses(base_dir, participant, run):
    target_run_number = int(re.match(r'run(.*)', run).group(1))
    folder = os.path.join(base_dir, participant, 'buttonpresses')

    # look for .mat file 
    for file in os.listdir(folder):
        match_mat = re.match(r'.*run(.*)_clicks\.mat$', file)
        if match_mat:
            run_number = int(match_mat.group(1))
            if run_number == target_run_number:
                return os.path.join(folder, file)
    
    # if no matching .mat files are found, look for text files
    for file in os.listdir(folder):
        match_txt = re.match(r'.*run(\d+)', file)
        if match_txt:
            run_number = int(match_txt.group(1))
            if run_number == target_run_number:
                return os.path.join(folder, file)

def get_rois(participant, run):
    base_dir = '/orcd/data/ldlewis/001/om2/ncicero/AAN'
    target_run_number = int(re.match(r'run(.*)', run).group(1))
    roi_folder = os.path.join(base_dir, participant, 'rois')
    rois = {}
    brainstem_new = {}

    for file in sorted(os.listdir(roi_folder)):
        match_roi = re.match(r"^(?!.*dilated_isolated)([^_]+(?:_new)?)_run(\d+)_nordic_stc_mc2_timecourse\.txt$", file)
        if match_roi:
            roi = match_roi.group(1)
            run_number = int(match_roi.group(2))
            if run_number == target_run_number:
                if 'new' in roi:
                    brainstem = roi.split('_')[0]
                    brainstem_new[brainstem] = os.path.join(roi_folder, file)
                else:
                    rois[roi] = os.path.join(roi_folder, file)
    rois.update(brainstem_new)
    return rois

def get_cortex_parcelations(participant, run):
    base_dir = '/orcd/data/ldlewis/001/om2/ncicero/AAN'
    target_run_number = int(re.match(r'run(.*)', run).group(1))
    cortex_folder = os.path.join(base_dir, participant, 'rois', 'ctx')
    rois = {}
    for file in sorted(os.listdir(cortex_folder)):
        roi = re.sub(r'^ctx-|_run.*$', '', file)
        run_number = int(re.search(r'_run(\d+)', file).group(1))
        if run_number == target_run_number:
            rois[roi] = os.path.join(cortex_folder, file)
    return rois

def get_thalamic_parcelations(participant, run):
    base_dir = '/orcd/data/ldlewis/001/om2/ncicero/AAN'
    target_run_number = int(re.match(r'run(.*)', run).group(1))
    thalamus_folder = os.path.join(base_dir, participant, 'rois', 'thal')
    rois = {}
    for file in sorted(os.listdir(thalamus_folder)):
        roi = re.sub(r'_run.*$', '', file)
        run_number = int(re.search(r'_run(\d+)', file).group(1))
        if run_number == target_run_number:
            rois[roi] = os.path.join(thalamus_folder, file)
    return rois

def get_mclogs(base_dir, participant, run):
    target_run_number = int(re.match(r'run(.*)', run).group(1))
    folder = os.path.join(base_dir, participant, 'stcfsl_mc2', 'mclogs')
    for file in os.listdir(folder):
        match = re.match(r'run(\d+).*\.mcdat$', file)
        if match:
            run_number = int(match.group(1))
            if run_number == target_run_number:
                return os.path.join(folder, file)
            
def copy_paste(src, participant, run, new_filename, subfolder=None):
    run_number = int(re.match(r'run(.*)', run).group(1))
    dest_folder = os.path.join(os.getcwd(), 'nicks_data', participant, f'run{run_number:02d}')
    if subfolder:
        dest_folder = os.path.join(dest_folder, subfolder)
    os.makedirs(dest_folder, exist_ok=True)
    shutil.copy2(src, os.path.join(dest_folder, new_filename))
 
def copy_nicks_data(mclogs=True, buttonpresses=True, rois=True, parcelate_cortex=True, parcelate_thalamus=True):
    base_dir = '/orcd/data/ldlewis/001/om2/ncicero/AAN'
    for participant in get_participants(base_dir):
        runs = get_runs(base_dir, participant)
        for run in runs:
            if mclogs:
                mclog_file = get_mclogs(base_dir, participant, run)
                copy_paste(mclog_file, participant, run, 'mclog.mcdat')
            if buttonpresses:
                target_bp_file = get_buttonpresses(base_dir, participant, run)
                if target_bp_file.endswith('.mat'):
                    copy_paste(target_bp_file, participant, run, 'bp.mat')
                else:
                    copy_paste(target_bp_file, participant, run, 'bp')
            if rois:
                rois_dict = get_rois(participant, run)
                for roi_name, roi_file in rois_dict.items():
                    copy_paste(roi_file, participant, run, f'{roi_name}.txt')
            if parcelate_cortex:
                cortex_rois_dict = get_cortex_parcelations(participant, run)
                for roi_name, roi_file in cortex_rois_dict.items():
                    copy_paste(roi_file, participant, run, f'{roi_name}.txt', subfolder='cortex')
            if parcelate_thalamus:
                thalamus_rois_dict = get_thalamic_parcelations(participant, run)
                for roi_name, roi_file in thalamus_rois_dict.items():
                    copy_paste(roi_file, participant, run, f'{roi_name}.txt', subfolder='thalamus')

def check_all_rois_present(all_rois):
    base_dir = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/nicks_data'
    for participant in sorted(os.listdir(base_dir)):
        for run in sorted(os.listdir(os.path.join(base_dir, participant))):
            rois_found = set()
            for file in os.listdir(os.path.join(base_dir, participant, run)):
                match_roi = re.match(r'^(.*)\.txt$', file)
                if match_roi:
                    rois_found.add(match_roi.group(1))
            missing_rois = set(all_rois) - rois_found
            if missing_rois:
                print(f'Participant: {participant}, Run: {run}, Missing ROIs: {sorted(list(missing_rois))}')

def check_all_cortex_parcelations_present():
    nicks_dir = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/nicks_data'
    all_cortex_rois = set()
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            cortex_folder = os.path.join(nicks_dir, participant, run, 'cortex')
            for roi_txt in os.listdir(cortex_folder):
                roi = roi_txt.removesuffix('.txt')
                all_cortex_rois.add(roi)
    
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            cortex_folder = os.path.join(nicks_dir, participant, run, 'cortex')
            rois_found = set()
            for roi_txt in os.listdir(cortex_folder):
                roi = roi_txt.removesuffix('.txt')
                rois_found.add(roi)
            missing_rois = all_cortex_rois - rois_found
            if missing_rois:
                print(f'Participant: {participant}, Run: {run}, Missing Cortex Parcelations: {sorted(list(missing_rois))}')

def check_all_thalamus_parcelations_present():
    nicks_dir = '/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/nicks_data'
    all_thalamus_rois = set()
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            thalamus_folder = os.path.join(nicks_dir, participant, run, 'thalamus')
            for roi_txt in os.listdir(thalamus_folder):
                roi = roi_txt.removesuffix('.txt')
                all_thalamus_rois.add(roi)
    
    for participant in sorted(os.listdir(nicks_dir)):
        for run in sorted(os.listdir(os.path.join(nicks_dir, participant))):
            thalamus_folder = os.path.join(nicks_dir, participant, run, 'thalamus')
            rois_found = set()
            for roi_txt in os.listdir(thalamus_folder):
                roi = roi_txt.removesuffix('.txt')
                rois_found.add(roi)
            missing_rois = all_thalamus_rois - rois_found
            if missing_rois:
                print(f'Participant: {participant}, Run: {run}, Missing Thalamus Parcelations: {sorted(list(missing_rois))}')

if __name__ == "__main__":
    copy_nicks_data(mclogs=False, buttonpresses=False, rois=False, parcelate_cortex=False, parcelate_thalamus=True)
    # check_all_cortex_parcelations_present()
    # check_all_thalamus_parcelations_present()
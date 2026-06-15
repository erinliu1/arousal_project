print('✅ connected', flush=True)

import os
print("✅ imported os", flush=True)

import numpy as np
print("✅ imported numpy", flush=True)

import pandas as pd
print("✅ imported pandas", flush=True)

from sklearn.linear_model import LogisticRegression
print("✅ imported LogisticRegression", flush=True)

from sklearn.pipeline import Pipeline
print("✅ imported Pipeline", flush=True)

from sklearn.preprocessing import StandardScaler
print("✅ imported StandardScaler", flush=True)

from dataloader import DataLoader
print("✅ imported DataLoader", flush=True)

class HyperparameterTuner:
  def __init__(self, bootstrap_seed, C, l1_ratio, solver, cv_leave_out='participant', model_type='flattened'):
    # cv_leave_out can be 'run' or 'participant'
    # model_type can be 'flattened' or 'snapshot'
    self.bootstrap_seed = bootstrap_seed
    self.C = C
    self.l1_ratio = l1_ratio
    self.solver = solver
    self.cv_leave_out = cv_leave_out
    self.model_type = model_type
    self.dataloader = DataLoader(roi='all', bootstrap_seed=self.bootstrap_seed, gap=0)
    self.dataset, self.loocv_lookup = self.dataloader.dataset, self.dataloader.loocv_lookup
    self.participants = sorted(self.dataset.keys())
    
  def process_X(self, X):
    if self.model_type == 'flattened':
      # flatten ROIs and TRs
      return X.reshape(X.shape[0], -1)
    elif self.model_type == 'snapshot':
      n_segments, n_rois, n_trs = X.shape
      X = np.transpose(X, (0, 2, 1)) # shape (n_segments, n_trs, n_rois)
      X = X.reshape(n_segments * n_trs, n_rois) # shape (n_segments * n_trs, n_rois)
      # tr_positions = np.arange(n_trs) / (n_trs - 1) # normalized within-segment TR position
      # tr_positions = np.tile(tr_positions, n_segments) # shape (n_segments * n_trs,)
      # tr_positions = tr_positions[:, np.newaxis] # shape (n_segments * n_trs, 1)
      # X = np.concatenate([X, tr_positions], axis=1) # shape (n_segments * n_trs, n_rois + 1)
      return X
    else:
      raise ValueError(f"model_type must be 'flattened' or 'snapshot', not {self.model_type}")

  def get_data(self, run_ids):
    X_list, y_list = [], []
    for i in run_ids:
        participant, run_id = self.loocv_lookup[i]
        run_data = self.dataset[participant][run_id]
        positives = run_data['positives']
        negatives = run_data['negatives']
        X_i = self.process_X(np.concatenate([positives, negatives], axis=0))

        y_i = np.array(run_data['true'])
        if self.model_type == 'snapshot':
          # repeat labels for each TR
          n_trs = positives.shape[2] # number of TRs per segment
          y_i = np.repeat(y_i, n_trs) # shape (n_segments * n_trs,)

        X_list.append(X_i)
        y_list.append(y_i)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y
  
  def fit_model(self, X_train, y_train):  
    max_iter = 10000 if self.solver == 'saga' else 1000
    clf = Pipeline([
      ('scaler', StandardScaler()),
      ('clf', LogisticRegression(
          C=self.C, 
          l1_ratio=self.l1_ratio,
          solver=self.solver, 
          random_state=self.bootstrap_seed,
          max_iter=max_iter))
    ])
    clf.fit(X_train, y_train)
    return clf

  def get_cv_folds(self):
    if self.cv_leave_out == 'run':
      for i in self.loocv_lookup:
        train_run_indices = [j for j in self.loocv_lookup if j != i]
        test_run_indices = [i]
        yield train_run_indices, test_run_indices
    elif self.cv_leave_out == 'participant':
      for participant in self.participants:
        train_run_indices = [i for i, (p, _) in self.loocv_lookup.items() if p != participant]
        test_run_indices = [i for i, (p, _) in self.loocv_lookup.items() if p == participant]
        yield train_run_indices, test_run_indices
    else:
      raise ValueError(f"Can only do CV leave out on 'run' or 'participant', not {self.cv_leave_out}")

  def run_LOOCV(self):
    accuracies = []
    for train_run_indices, test_run_indices in self.get_cv_folds():
      X_train, y_train = self.get_data(train_run_indices)
      clf = self.fit_model(X_train, y_train)
      for i in test_run_indices:
        participant, run_id = self.loocv_lookup[i]
        X_test, y_test = self.get_data([i])
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] # probability of positive class
        accuracy = np.mean(y_pred == y_test)
        run_results = {
            'bootstrap_seed': self.bootstrap_seed,
            'participant': participant,
            'run_id': run_id,
            'C': self.C,
            'l1_ratio': self.l1_ratio,
            'solver': self.solver,
            'accuracy': accuracy,
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist(),
            'y_true': y_test.tolist(),
        }       
        if self.model_type == 'snapshot':
          run_data = self.dataset[participant][run_id]
          n_segments = len(run_data['true'])
          n_trs = run_data['positives'].shape[2]
          tr_positions = np.tile(np.arange(n_trs), n_segments) 
          run_results['tr_positions'] = tr_positions.tolist()
        accuracies.append(run_results)
        
    return accuracies

param_lookup = {
    'l2': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'l1_ratio': [0.0],
        'solver': ['lbfgs', 'liblinear']
    },
    'elastic': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9],
        'solver': ['saga']
    }
}

slurm_lookup = {}
i = 0
for loocv in ['lopocv', 'lorocv']:
  for bootstrap_seed in range(20):
    for C in param_lookup['l2']['C']:
      for l1_ratio in param_lookup['l2']['l1_ratio']:
        for solver in param_lookup['l2']['solver']:
          slurm_lookup[i] = {'loocv': loocv, 'bootstrap_seed': bootstrap_seed, 'C': C, 'l1_ratio': l1_ratio, 'solver': solver}
          i += 1
    for C in param_lookup['elastic']['C']:
      for l1_ratio in param_lookup['elastic']['l1_ratio']:
        for solver in param_lookup['elastic']['solver']:
          slurm_lookup[i] = {'loocv': loocv, 'bootstrap_seed': bootstrap_seed, 'C': C, 'l1_ratio': l1_ratio, 'solver': solver}
          i += 1

# slurm_lookup is from 0 to 1399

model_type = 'snapshot'

array_id = int(os.environ['SLURM_ARRAY_TASK_ID']) 
params = slurm_lookup[array_id]
loocv = params['loocv']
bootstrap_seed = params['bootstrap_seed']
C = params['C']
l1_ratio = params['l1_ratio']
solver = params['solver']


save_dir = f'/orcd/data/ldlewis/001/om/erinliu/Arousal_Project/{model_type}_45/{loocv}/hyperparam_tuning_results'
os.makedirs(save_dir, exist_ok=True)

print(
    f'✅ beginning tuning for loocv={loocv}, bootstrap_seed={bootstrap_seed}, '
    f'C={C}, l1_ratio={l1_ratio}, solver={solver}',
    flush=True
)

if loocv == 'lopocv':
  cv_leave_out = 'participant'
elif loocv == 'lorocv':
  cv_leave_out = 'run'
else:
  print(f'❌ invalid loocv value: {loocv}', flush=True)

hyperparameter_tuner = HyperparameterTuner(bootstrap_seed=bootstrap_seed, C=C, l1_ratio=l1_ratio, solver=solver, cv_leave_out=cv_leave_out, model_type=model_type)
result = hyperparameter_tuner.run_LOOCV()
results_df = pd.DataFrame(result)
save_path = f'{save_dir}/{array_id}.csv'
results_df.to_csv(save_path, index=False) 

print(f'✅ results saved to {save_path}', flush=True)
"""Reproduce the epoch21 comparison from aligned probabilities, without text or training."""
from pathlib import Path
import json
import numpy as np
from backend.src.alpha_tuning import tune_alpha, compute_ensemble_probabilities, evaluate_probabilities
from backend.src.max_confidence import select_max_confidence


def main():
    root=Path(__file__).resolve().parents[1]/'results/epoch21'
    # Validation selection is completed before opening the Test cache.
    with np.load(root/'val_aligned_probabilities.npz',allow_pickle=False) as saved:
        val={k:saved[k].copy() for k in saved.files}
    selected=tune_alpha(val['y_true'],val['svm_probs'],val['transformer_probs'])
    alpha=selected['selected_alpha']
    splits={'val':val}
    with np.load(root/'test_aligned_probabilities.npz',allow_pickle=False) as saved:
        splits['test']={k:saved[k].copy() for k in saved.files}
    result={'selected_alpha_from_validation':alpha,'splits':{}}
    for name,data in splits.items():
        y,svm,tf=data['y_true'],data['svm_probs'],data['transformer_probs']
        probabilities={'svm':svm,'transformer':tf,
            'max_confidence':select_max_confidence(svm,tf)['probabilities'],
            'ensemble':compute_ensemble_probabilities(svm,tf,alpha)}
        result['splits'][name]={method:evaluate_probabilities(y,p) for method,p in probabilities.items()}
    expected=json.loads((root/'results.json').read_text(encoding='utf-8'))
    assert alpha==expected['selected_alpha']
    for name,models in result['splits'].items():
        for method,metrics in models.items():
            for metric,value in metrics.items():
                assert abs(value-expected['splits'][name]['models'][method][metric])<1e-12,(name,method,metric)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('All four methods match the saved results; model/data/deployment files were not changed.')


if __name__=='__main__':main()

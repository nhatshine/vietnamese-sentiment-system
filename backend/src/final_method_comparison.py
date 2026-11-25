"""Fixed Test comparison and paired uncertainty; no model loading or tuning."""
import numpy as np
from scipy.stats import binomtest

from .alpha_tuning import _labels, _probabilities, compute_ensemble_probabilities
from .config import LABELS
from .error_analysis import SOURCE_NAMES, confidence_analysis, model_metrics
from .max_confidence import FIXED_ALPHA, MODEL_TIE_RULE, CLASS_TIE_RULE, select_max_confidence


def assert_ensemble_baseline(actual, baseline):
    for key in ('accuracy','macro_precision','macro_recall','macro_f1'):
        if not np.isclose(actual[key],baseline[key],rtol=0,atol=1e-12):
            raise ValueError(f'Ensemble {key} differs from the frozen Test baseline; stop.')
    if actual['confusion_matrix'] != baseline['confusion_matrix']:
        raise ValueError('Ensemble confusion matrix differs from historical Test results.')
    if 'per_class' in baseline:
        for a,b in zip(actual['per_class'],baseline['per_class'],strict=True):
            if a['label']!=b['label'] or a['support']!=b['support']:
                raise ValueError('Ensemble historical per-class alignment mismatch.')
            if any(not np.isclose(a[k],b[k],rtol=0,atol=1e-12) for k in ('precision','recall','f1')):
                raise ValueError('Ensemble historical per-class scores mismatch.')


def evaluate_fixed_methods(y_true,svm_probs,transformer_probs,sources,ensemble_baseline,*,label_order=LABELS):
    chosen=select_max_confidence(svm_probs,transformer_probs,label_order=label_order)
    truth=_labels(y_true,len(chosen['probabilities']))
    source_values=np.asarray(sources)
    if source_values.ndim!=1 or len(source_values)!=len(truth) or not np.isin(source_values,SOURCE_NAMES).all():
        raise ValueError('Test source/sample alignment mismatch.')
    matrices={'max_confidence':chosen['probabilities'],
              'weighted_ensemble':compute_ensemble_probabilities(svm_probs,transformer_probs,FIXED_ALPHA)}
    models={name:model_metrics(truth,values) for name,values in matrices.items()}
    assert_ensemble_baseline(models['weighted_ensemble'],ensemble_baseline)
    diagnostics={name:confidence_analysis(truth,values,bins=10) for name,values in matrices.items()}
    for name,model in models.items():
        model.update(ece=diagnostics[name]['ece'],mean_confidence=diagnostics[name]['mean_confidence'],
                     wrong_high_confidence_count=diagnostics[name]['wrong_high_confidence_counts']['0.9'])
    mc=matrices['max_confidence'].argmax(1);we=matrices['weighted_ensemble'].argmax(1)
    a,b=mc==truth,we==truth
    by_source=[]
    for source in SOURCE_NAMES:
        mask=source_values==source
        if mask.any():
            by_source.append({'source':source,'sample_count':int(mask.sum()),
                              'models':{name:{**model_metrics(truth[mask],v[mask]),
                                        'confidence':confidence_analysis(truth[mask],v[mask],bins=10)}
                                        for name,v in matrices.items()}})
    delta=models['weighted_ensemble']['macro_f1']-models['max_confidence']['macro_f1']
    return {'evaluation_dataset':'test','sample_count':len(truth),'alpha':FIXED_ALPHA,'label_order':list(LABELS),
            'model_tie_rule':MODEL_TIE_RULE,'class_tie_rule':CLASS_TIE_RULE,'models':models,
            'by_source':by_source,'confidence_diagnostics':diagnostics,
            'selection_counts':{'svm':int((chosen['selected_models']=='svm').sum()),
                                'transformer':int((chosen['selected_models']=='transformer').sum()),
                                'exact_ties':int(chosen['ties'].sum())},
            'base_mean_confidence':{'svm':float(_probabilities(svm_probs,'svm').max(1).mean()),
                                    'transformer':float(_probabilities(transformer_probs,'transformer').max(1).mean())},
            'correctness_groups':{'both_correct':int((a&b).sum()),'max_confidence_only':int((a&~b).sum()),
                                  'weighted_ensemble_only':int((~a&b).sum()),'both_wrong':int((~a&~b).sum())},
            'disagreements':{'count':int((mc!=we).sum()),'indices':np.flatnonzero(mc!=we).tolist(),
                             'both_wrong_count':int(((mc!=we)&~a&~b).sum())},
            'macro_f1_delta_weighted_minus_max':float(delta),
            'observed_winner_on_test':'weighted_ensemble' if delta>0 else 'max_confidence' if delta<0 else 'tie',
            'baseline_verified':True,'alpha_retuned':False,'test_based_method_selection':False}


def _macro_f1(matrix):
    denominator=matrix.sum(0)+matrix.sum(1)
    return float(np.divide(2*np.diag(matrix),denominator,out=np.zeros(3,dtype=float),where=denominator!=0).mean())


def paired_statistics(y_true,max_predictions,weighted_predictions,*,replicates=10000,seed=42):
    """McNemar tests correctness; paired percentile bootstrap estimates delta F1."""
    truth=np.asarray(y_true)
    truth=_labels(truth,len(truth))
    if not len(truth): raise ValueError('Empty Test sample.')
    mc=_labels(max_predictions,len(truth));we=_labels(weighted_predictions,len(truth))
    if isinstance(replicates,bool) or not isinstance(replicates,int) or replicates<2:
        raise ValueError('replicates must be an integer >=2.')
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:
        raise ValueError('seed must be a nonnegative integer.')
    a,b=mc==truth,we==truth
    max_only=int((a&~b).sum());weighted_only=int((~a&b).sum());discordant=max_only+weighted_only
    p=float(binomtest(max_only,discordant,.5,alternative='two-sided').pvalue) if discordant else 1.
    def delta_at(indices):
        joint=np.bincount(9*truth[indices]+3*mc[indices]+we[indices],minlength=27).reshape(3,3,3)
        return _macro_f1(joint.sum(axis=1))-_macro_f1(joint.sum(axis=2))
    observed=delta_at(np.arange(len(truth)))
    rng=np.random.default_rng(seed);deltas=np.empty(replicates)
    for i in range(replicates):
        indices=rng.integers(0,len(truth),size=len(truth))
        deltas[i]=delta_at(indices)
    ci=np.quantile(deltas,[.025,.975],method='linear')
    return {'mcnemar':{'method':'exact two-sided binomial McNemar','max_only_correct':max_only,
                      'weighted_only_correct':weighted_only,'discordant_count':discordant,'p_value':p,
                      'significance_level':.05,'reject_equal_correctness_at_0_05':p<.05,
                      'target':'paired correctness / accuracy, not Macro-F1'},
            'paired_bootstrap':{'method':'IID paired row percentile bootstrap','replicates':replicates,'seed':seed,
                                'sample_count':len(truth),'same_sample_indices_for_both_methods':True,
                                'label_order':list(LABELS),'observed_delta':observed,'delta_definition':'Weighted Macro-F1 minus Max Confidence Macro-F1',
                                'ci_95':ci.tolist(),'standard_error':float(deltas.std(ddof=1)),
                                'confidence_level':.95,'quantile_method':'linear',
                                'ci_excludes_zero':bool(ci[0]>0 or ci[1]<0),
                                'limitations':'Fixed trained models and pooled Test rows assumed IID. Does not include training randomness, dataset shift or dependence within source/user/thread.'}}

"""Step 14.2: fixed rules, hand calculations and paired statistics."""
import copy
import unittest
import numpy as np

try:
    from backend.src.final_method_comparison import evaluate_fixed_methods, paired_statistics
except ModuleNotFoundError as error:
    if error.name != 'backend.src.final_method_comparison':
        raise
    evaluate_fixed_methods = paired_statistics = None


class FixedTestComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(evaluate_fixed_methods, 'Test comparison not implemented')
        self.truth = np.array([0, 2, 0, 1])
        self.svm = np.array([[.8,.1,.1],[.6,.2,.2],[.6,.2,.2],[.8,.1,.1]])
        self.tf = np.array([[.7,.2,.1],[.05,.05,.9],[.05,.05,.9],[.7,.2,.1]])
        self.sources = ['hieu_sa','hieu_sa','neu_esc','vsfc']
        self.baseline = {'accuracy':.5,'macro_precision':1/6,'macro_recall':1/3,'macro_f1':2/9,
                         'confusion_matrix':[[2,0,0],[1,0,0],[1,0,0]]}

    def run_comparison(self, **changes):
        args = dict(y_true=self.truth,svm_probs=self.svm,transformer_probs=self.tf,
                    sources=self.sources,ensemble_baseline=self.baseline)
        args.update(changes)
        return evaluate_fixed_methods(**args)

    def test_test_metrics_and_fixed_alpha_match_hand_calculations(self):
        result = self.run_comparison()
        self.assertEqual(result['evaluation_dataset'], 'test')
        self.assertEqual(result['alpha'], .8)
        self.assertAlmostEqual(result['models']['max_confidence']['macro_f1'],7/18)
        self.assertAlmostEqual(result['models']['weighted_ensemble']['macro_f1'],2/9)
        self.assertEqual(result['models']['max_confidence']['confusion_matrix'],[[1,0,1],[1,0,0],[0,0,1]])
        self.assertNotIn('winner_on_validation',result)

    def test_counts_partition_samples_and_disagreements(self):
        r=self.run_comparison()
        self.assertEqual(r['correctness_groups'],{'both_correct':1,'max_confidence_only':1,'weighted_ensemble_only':1,'both_wrong':1})
        self.assertEqual(r['disagreements']['indices'],[1,2])
        self.assertEqual(r['selection_counts'],{'svm':2,'transformer':2,'exact_ties':0})

    def test_existing_baseline_mismatch_stops_evaluation(self):
        for key,value in [('macro_f1',.99),('confusion_matrix',[[1,1,0],[1,0,0],[1,0,0]])]:
            bad=copy.deepcopy(self.baseline);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.run_comparison(ensemble_baseline=bad)

    def test_cannot_override_alpha_or_label_order(self):
        with self.assertRaises(TypeError):
            self.run_comparison(alpha=.7)
        with self.assertRaises(ValueError):
            self.run_comparison(label_order=['positive','neutral','negative'])

    def test_does_not_modify_input_arrays(self):
        a,b=self.svm.copy(),self.tf.copy(); self.run_comparison()
        np.testing.assert_array_equal(self.svm,a);np.testing.assert_array_equal(self.tf,b)

    def test_exact_confidence_tie_prefers_svm_in_test_evaluation(self):
        self.tf[0]=[.1,.8,.1]
        r=self.run_comparison()
        self.assertEqual(r['selection_counts']['exact_ties'],1)
        self.assertEqual(r['models']['max_confidence']['confusion_matrix'],[[1,0,1],[1,0,0],[0,0,1]])

    def test_invalid_probabilities_and_alignment_are_rejected(self):
        for changes in ({'svm_probs':self.svm[:2]},{'sources':self.sources[:2]},
                        {'y_true':[0,2,0,3]},{'svm_probs':np.ones((4,3))},
                        {'svm_probs':np.full((4,3),np.nan)}):
            with self.subTest(changes=changes),self.assertRaises((TypeError,ValueError)):
                self.run_comparison(**changes)

    def test_source_metrics_and_ece_keep_the_same_rows(self):
        r=self.run_comparison()
        self.assertEqual([x['sample_count'] for x in r['by_source']],[2,1,1])
        self.assertAlmostEqual(r['confidence_diagnostics']['max_confidence']['ece'],.35)
        self.assertEqual(r['confidence_diagnostics']['max_confidence']['wrong_high_confidence_counts']['0.9'],1)
        self.assertEqual([x['label'] for x in r['models']['max_confidence']['per_class']],['negative','neutral','positive'])


class PairedStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(paired_statistics,'Paired statistics not implemented')

    def test_identical_predictions_have_zero_delta_and_unit_p(self):
        r=paired_statistics([0,1,2],[0,1,2],[0,1,2],replicates=80,seed=42)
        self.assertEqual(r['mcnemar']['p_value'],1.)
        self.assertEqual(r['paired_bootstrap']['ci_95'],[0.,0.])

    def test_exact_mcnemar_matches_binomial_hand_calculation(self):
        # Six discordant samples, all favor Weighted: 2*(1/2)^6 = 1/32.
        r=paired_statistics([0]*6,[1]*6,[0]*6,replicates=20,seed=42)
        self.assertEqual(r['mcnemar']['max_only_correct'],0)
        self.assertEqual(r['mcnemar']['weighted_only_correct'],6)
        self.assertEqual(r['mcnemar']['p_value'],1/32)
        self.assertAlmostEqual(r['paired_bootstrap']['observed_delta'],1/3)

    def test_bootstrap_uses_pairs_and_is_reproducible(self):
        a=paired_statistics([0,1,2,0],[0,1,2,0],[1,1,2,0],replicates=200,seed=91)
        b=paired_statistics([0,1,2,0],[0,1,2,0],[1,1,2,0],replicates=200,seed=91)
        self.assertEqual(a,b)
        self.assertTrue(a['paired_bootstrap']['same_sample_indices_for_both_methods'])
        self.assertLessEqual(a['paired_bootstrap']['ci_95'][1],0)

    def test_fixed_three_class_macro_in_bootstrap_with_absent_class(self):
        r=paired_statistics([0],[1],[0],replicates=20,seed=42)
        np.testing.assert_allclose(r['paired_bootstrap']['ci_95'],[1/3,1/3])

    def test_bootstrap_matches_independent_sklearn_replicates(self):
        from sklearn.metrics import f1_score
        y=np.array([0,1,2,0]);a=np.array([0,1,1,0]);b=np.array([0,1,2,1])
        rng=np.random.default_rng(42);d=[]
        for _ in range(30):
            idx=rng.integers(0,4,size=4)
            d.append(f1_score(y[idx],b[idx],labels=[0,1,2],average='macro',zero_division=0)-
                     f1_score(y[idx],a[idx],labels=[0,1,2],average='macro',zero_division=0))
        r=paired_statistics(y,a,b,replicates=30,seed=42)
        np.testing.assert_allclose(r['paired_bootstrap']['ci_95'],np.quantile(d,[.025,.975]),atol=1e-15)

    def test_statistics_reject_bad_sample_alignment_or_parameters(self):
        for args in (([0],[0,1],[0]),([0],[3],[0])):
            with self.assertRaises(ValueError): paired_statistics(*args,replicates=20,seed=42)
        for n in (0,True,2.5):
            with self.assertRaises((TypeError,ValueError)): paired_statistics([0],[0],[0],replicates=n,seed=42)


if __name__=='__main__': unittest.main()

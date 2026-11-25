"""Synthetic tests for frozen Test evaluation; no checkpoint is required."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from backend.src.config import LABELS, LABEL_TO_ID
from backend.src.final_evaluation import (
    evaluate_test, load_frozen_selection, load_test_dataset, ordered_row_ids, prediction_rows,
)


def selection():
    return {"selection_dataset": "validation", "test_dataset_used": False,
            "primary_metric": "macro_f1", "selected_alpha": .8,
            "selected_validation_macro_f1": .83, "svm_validation_macro_f1": .82,
            "transformer_validation_macro_f1": .79, "ensemble_validation_macro_f1": .83,
            "probability_columns": LABELS, "label_mapping": LABEL_TO_ID,
            "fingerprints": {"model": "abc"}, "sample_count": 8238}


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "selection.json"
        self.path.write_text(json.dumps(selection()), encoding="utf-8")
        self.frozen = load_frozen_selection(self.path)
        self.truth = np.array([0, 1, 2, 0])
        self.svm = np.array([[.8,.1,.1],[.2,.7,.1],[.1,.2,.7],[.2,.6,.2]])
        self.transformer = np.array([[.1,.8,.1],[.2,.7,.1],[.1,.8,.1],[.7,.2,.1]])
        self.sources = ["hieu_sa", "neu_esc", "vsfc", "hieu_sa"]

    def tearDown(self):
        self.temp.cleanup()

    def evaluate(self, **changes):
        args = dict(y_true=self.truth, svm_probs=self.svm, transformer_probs=self.transformer,
                    sources=self.sources, frozen=self.frozen)
        args.update(changes)
        return evaluate_test(**args)

    def rewrite(self, **changes):
        value = selection()
        value.update(changes)
        self.path.write_text(json.dumps(value), encoding="utf-8")
        return load_frozen_selection(self.path)

    def test_selection_reads_alpha_and_validation_winner(self):
        self.assertEqual(self.frozen["selected_alpha"], .8)
        self.assertEqual(self.frozen["selected_method"], "ensemble")
        self.assertEqual(self.frozen["selection_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_selection_honors_artifact_not_prompt_constants(self):
        value = self.rewrite(selected_alpha=.7, svm_validation_macro_f1=.9)
        self.assertEqual(value["selected_alpha"], .7)
        self.assertEqual(value["selected_method"], "svm")

    def test_selection_rejects_test_based_selection(self):
        for changes in ({"selection_dataset":"test"}, {"test_dataset_used":True}, {"primary_metric":"accuracy"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.rewrite(**changes)

    def test_selection_rejects_mapping_and_inconsistent_score(self):
        for changes in ({"probability_columns":list(reversed(LABELS))},
                        {"label_mapping":{"negative":2,"neutral":1,"positive":0}},
                        {"selected_validation_macro_f1":.5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.rewrite(**changes)

    def test_selection_rejects_invalid_alpha_or_metric(self):
        for changes in ({"selected_alpha":True},{"selected_alpha":1.1},
                        {"selected_alpha":float('nan')},{"svm_validation_macro_f1":float('inf')}):
            with self.subTest(changes=changes), self.assertRaises((ValueError,TypeError)):
                self.rewrite(**changes)

    def test_metrics_are_from_whole_dataset(self):
        result, matrices = self.evaluate()
        m=result["models"]["svm"]
        self.assertAlmostEqual(m["accuracy"], .75)
        self.assertAlmostEqual(m["macro_precision"], 5/6)
        self.assertAlmostEqual(m["macro_recall"], 5/6)
        self.assertAlmostEqual(m["macro_f1"], 7/9)
        self.assertAlmostEqual(m["weighted_f1"], .75)
        self.assertEqual(m["confusion_matrix"], [[1,1,0],[0,1,0],[0,0,1]])
        self.assertEqual([r["support"] for r in m["per_class"]],[2,1,1])
        self.assertAlmostEqual(m["per_class"][0]["f1"], 2/3)
        self.assertAlmostEqual(result["models"]["transformer"]["accuracy"], .5)
        np.testing.assert_allclose(matrices["ensemble"], [[.66,.24,.1],[.2,.7,.1],[.1,.32,.58],[.3,.52,.18]])

    def test_source_filtering_and_fixed_three_class_macro(self):
        result,_=self.evaluate()
        h=next(r for r in result["sources"] if r["source"]=="hieu_sa")
        self.assertEqual(h["n"],2)
        self.assertEqual(h["label_counts"],{"negative":2,"neutral":0,"positive":0})
        self.assertAlmostEqual(h["models"]["svm"]["macro_f1"],2/9)
        self.assertEqual(sum(r["n"] for r in result["sources"]),4)

    def test_validation_test_delta_and_selection_survive_test_ranking(self):
        # At alpha0, Ensemble equals weaker Transformer; locked choice still Ensemble.
        frozen=copy.deepcopy(self.frozen)
        frozen["selected_alpha"]=0.
        result,_=self.evaluate(frozen=frozen)
        self.assertEqual(result["selected_method"],"ensemble")
        self.assertEqual(result["selected_alpha"],0.)
        self.assertGreater(result["models"]["svm"]["macro_f1"],result["models"]["ensemble"]["macro_f1"])
        self.assertLess(result["ensemble_deltas"]["vs_svm"],0)
        self.assertAlmostEqual(result["validation_vs_test"][0]["test_minus_validation"],7/9-.82)

    def test_no_mutation(self):
        a,b,f=self.svm.copy(),self.transformer.copy(),copy.deepcopy(self.frozen)
        self.evaluate()
        np.testing.assert_array_equal(a,self.svm)
        np.testing.assert_array_equal(b,self.transformer)
        self.assertEqual(f,self.frozen)

    def test_bad_probability_inputs(self):
        bad=[self.svm[:2],np.ones((4,2)),np.ones((4,3)),self.svm.copy(),self.svm.copy()]
        bad[-2][0,0]=np.nan
        bad[-1][0,0]=np.inf
        for value in bad:
            with self.subTest(shape=value.shape), self.assertRaises((ValueError,TypeError)):
                self.evaluate(svm_probs=value)

    def test_bad_label_and_source_alignment(self):
        for args in ({"y_true":self.truth[:3]}, {"y_true":np.array([0,1,2,3])},
                     {"sources":self.sources[:3]}, {"sources":["unknown"]*4}):
            with self.subTest(args=args), self.assertRaises((ValueError,TypeError)):
                self.evaluate(**args)

    def write_dataset(self, rows=None, headers=("text","label","source")):
        path=self.root/"test.csv"
        if rows is None:
            rows=[['  tiếng Việt, giữ nguyên  ',0,'hieu_sa'],['trung tính',1,'neu_esc'],['tốt',2,'vsfc']]
        with path.open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f); w.writerow(headers); w.writerows(rows)
        return path,hashlib.sha256(path.read_bytes()).hexdigest()

    def test_dataset_preserves_text_order_and_metadata(self):
        path,sha=self.write_dataset()
        d=load_test_dataset(path,sha,expected_rows=3)
        self.assertEqual(d['texts'],['  tiếng Việt, giữ nguyên  ','trung tính','tốt'])
        self.assertEqual(d['labels'].tolist(),[0,1,2])
        self.assertEqual(d['source_counts'],dict(hieu_sa=1,neu_esc=1,vsfc=1))

    def test_dataset_rejects_hash_rowcount_and_missing_file(self):
        path,sha=self.write_dataset()
        with self.assertRaises(ValueError): load_test_dataset(path,'bad',expected_rows=3)
        with self.assertRaises(ValueError): load_test_dataset(path,sha,expected_rows=4)
        with self.assertRaises(FileNotFoundError): load_test_dataset(self.root/'missing.csv',sha)

    def test_dataset_rejects_null_invalid_label_source_and_headers(self):
        for row in (['',0,'hieu_sa'],['x','', 'hieu_sa'],['x',3,'hieu_sa'],['x',0,'wrong']):
            path,sha=self.write_dataset([row])
            with self.subTest(row=row),self.assertRaises(ValueError): load_test_dataset(path,sha,expected_rows=1)
        path,sha=self.write_dataset([['x',0]],headers=('text','label'))
        with self.assertRaises(ValueError): load_test_dataset(path,sha,expected_rows=1)

    def test_row_ids_bind_text_label_source_and_order(self):
        d={'texts':['same','different'],'labels':np.array([0,0]),'sources':['hieu_sa']*2}
        ids=ordered_row_ids(d)
        self.assertNotEqual(ids[0],ids[1])
        changed=copy.deepcopy(d); changed['texts'].reverse()
        self.assertNotEqual(ids,ordered_row_ids(changed))
        changed=copy.deepcopy(d); changed['labels'][0]=1
        self.assertNotEqual(ids,ordered_row_ids(changed))

    def test_prediction_rows_have_exact_values_and_mapping(self):
        _,matrices=self.evaluate()
        d={'texts':['a','b','c','d'],'labels':self.truth,'sources':self.sources}
        rows=prediction_rows(d,matrices)
        self.assertEqual([r['sample_index'] for r in rows],[0,1,2,3])
        self.assertEqual(rows[3]['true_label'],'negative')
        self.assertEqual(rows[3]['svm_pred'],'neutral')
        self.assertEqual(rows[3]['transformer_pred'],'negative')
        self.assertAlmostEqual(rows[0]['ensemble_p_negative'],.66)
        self.assertEqual(rows[0]['svm_confidence'],.8)
        self.assertEqual(len(rows[0]),19)


class FinalRunnerTests(unittest.TestCase):
    setUp = FinalEvaluationTests.setUp
    tearDown = FinalEvaluationTests.tearDown
    evaluate = FinalEvaluationTests.evaluate
    write_dataset = FinalEvaluationTests.write_dataset
    def test_handoff_verification_copies_exact_bytes(self):
        import zipfile
        from backend.run_final_evaluation import prepare_test_dataset
        source,sha=self.write_dataset()
        archive=self.root/'handoff.zip'
        with zipfile.ZipFile(archive,'w') as z: z.writestr('data/test.csv',source.read_bytes())
        target=self.root/'project'/'test.csv'
        data,provenance=prepare_test_dataset(target,source,archive,sha,expected_rows=3)
        self.assertEqual(target.read_bytes(),source.read_bytes())
        self.assertEqual(data['sample_count'],3)
        self.assertEqual(provenance['transformer_archive_test_sha256'],sha)

    def test_handoff_mismatch_stops_without_copy(self):
        import zipfile
        from backend.run_final_evaluation import prepare_test_dataset
        source,sha=self.write_dataset()
        archive=self.root/'bad.zip'
        with zipfile.ZipFile(archive,'w') as z: z.writestr('data/test.csv',b'different')
        target=self.root/'project'/'test.csv'
        with self.assertRaises(ValueError): prepare_test_dataset(target,source,archive,sha,expected_rows=3)
        self.assertFalse(target.exists())

    def test_existing_test_mismatch_is_not_overwritten(self):
        import zipfile
        from backend.run_final_evaluation import prepare_test_dataset
        source,sha=self.write_dataset()
        archive=self.root/'handoff.zip'
        with zipfile.ZipFile(archive,'w') as z: z.writestr('data/test.csv',source.read_bytes())
        target=self.root/'other.csv'; target.write_bytes(b'wrong')
        with self.assertRaises(ValueError): prepare_test_dataset(target,source,archive,sha,expected_rows=3)
        self.assertEqual(target.read_bytes(),b'wrong')

    def cache_data(self):
        return {'texts':['a','b','c','d'],'labels':self.truth,'sources':self.sources,'sha256':'dataset'}

    def test_freeze_precedes_single_inference_and_second_run_reuses_cache(self):
        from backend.run_final_evaluation import load_or_predict
        data=self.cache_data(); output=self.root/'outputs'; calls=[]
        def infer(texts):
            lock=json.loads((output/'evaluation_freeze.json').read_text())
            self.assertEqual(lock['configuration']['selected_method'],'ensemble')
            calls.append(list(texts))
            return self.svm,self.transformer
        with patch('backend.run_final_evaluation.infer_cleaned',side_effect=infer):
            a,b,status=load_or_predict(data,self.frozen,output)
            a2,b2,status2=load_or_predict(data,self.frozen,output)
        self.assertEqual(len(calls),1)
        self.assertEqual(status,'computed_once')
        self.assertEqual(status2,'reused')
        np.testing.assert_array_equal(a,a2)
        np.testing.assert_array_equal(b,b2)

    def test_cache_rejects_changed_rows_or_frozen_alpha(self):
        from backend.run_final_evaluation import load_or_predict
        data=self.cache_data(); output=self.root/'outputs'
        with patch('backend.run_final_evaluation.infer_cleaned',return_value=(self.svm,self.transformer)):
            load_or_predict(data,self.frozen,output)
        changed=copy.deepcopy(data); changed['texts'].reverse()
        with self.assertRaises(ValueError): load_or_predict(changed,self.frozen,output)
        changed=copy.deepcopy(self.frozen); changed['selected_alpha']=.5
        with self.assertRaises(ValueError): load_or_predict(data,changed,output)

    def test_cache_corruption_stops_instead_of_inference_again(self):
        from backend.run_final_evaluation import load_or_predict
        data=self.cache_data(); output=self.root/'outputs'
        with patch('backend.run_final_evaluation.infer_cleaned',return_value=(self.svm,self.transformer)):
            load_or_predict(data,self.frozen,output)
        (output/'test_probabilities.npz').write_bytes(b'corrupt')
        with self.assertRaises(ValueError): load_or_predict(data,self.frozen,output)

    def test_cleaned_apis_receive_text_once_and_raw_preprocess_is_not_used(self):
        from backend.run_final_evaluation import infer_cleaned
        from backend.src import real_svm,real_transformer
        calls=[]
        def predict(texts,**kwargs):
            calls.append(list(texts)); return [{label:float(self.svm[i,j]) for j,label in enumerate(LABELS)} for i in range(len(texts))]
        with patch.object(real_svm,'predict_svm_cleaned_batch',side_effect=predict), \
             patch.object(real_transformer,'predict_transformer_cleaned_batch',side_effect=predict), \
             patch.object(real_svm,'predict_svm',side_effect=AssertionError('raw predictor forbidden')), \
             patch.object(real_transformer,'predict_transformer',side_effect=AssertionError('raw predictor forbidden')):
            a,b=infer_cleaned(['cleaned']*4)
        self.assertEqual(calls,[['cleaned']*4,['cleaned']*4])
        np.testing.assert_array_equal(a,self.svm)
        np.testing.assert_array_equal(b,self.svm)

    def test_export_csv_json_report_and_plots_consistent(self):
        from backend.run_final_evaluation import export_results
        result,matrices=self.evaluate()
        result['dataset']={'sample_count':4,'sha256':'x','label_counts':{'negative':2,'neutral':1,'positive':1},'source_counts':{'hieu_sa':2,'neu_esc':1,'vsfc':1}}
        result['provenance']={'cache_status':'synthetic','selected_before_test_inference':True}
        report=self.root/'report.md'; output=self.root/'outputs'
        export_results(result,self.cache_data(),matrices,output,report)
        saved=json.loads((output/'test_metrics.json').read_text(encoding='utf-8'))
        with (output/'test_predictions.csv').open(encoding='utf-8',newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),4)
        self.assertAlmostEqual(float(rows[0]['ensemble_p_negative']),.66)
        self.assertEqual(rows[3]['svm_pred'],'neutral')
        self.assertEqual(saved['selected_method'],'ensemble')
        self.assertAlmostEqual(saved['models']['svm']['macro_f1'],7/9)
        self.assertEqual(len(list(output.glob('*.png'))),8)
        self.assertTrue(report.stat().st_size>1000)

    def test_posthoc_handoff_comparison_does_not_change_selection(self):
        import zipfile
        from backend.run_final_evaluation import compare_handoff_results
        result,_=self.evaluate()
        before=copy.deepcopy(result)
        meta=self.root/'meta.json'
        meta.write_text(json.dumps({'test':{'overall':{'accuracy':.75,'macro_f1':7/9}}}))
        archive=self.root/'handoff.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('transformer/metrics/metrics.json',json.dumps({'test':{'accuracy':.5,'macro_f1':result['models']['transformer']['macro_f1']},'split_sizes':{'train':65899,'val':8238,'test':8238}}))
        checks=compare_handoff_results(result,meta,archive)
        self.assertTrue(checks['models']['svm']['matches_within_tolerance'])
        self.assertTrue(checks['models']['transformer']['matches_within_tolerance'])
        self.assertEqual(result,before)

    def test_posthoc_mismatch_is_reported_without_reselection(self):
        import zipfile
        from backend.run_final_evaluation import compare_handoff_results
        result,_=self.evaluate()
        meta=self.root/'meta.json'
        meta.write_text(json.dumps({'test':{'overall':{'accuracy':0.,'macro_f1':0.}}}))
        archive=self.root/'handoff.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('transformer/metrics/metrics.json',json.dumps({'test':{'accuracy':0.,'macro_f1':0.}}))
        checks=compare_handoff_results(result,meta,archive)
        self.assertFalse(checks['models']['svm']['matches_within_tolerance'])
        self.assertEqual(result['selected_method'],'ensemble')


if __name__=='__main__':
    unittest.main()

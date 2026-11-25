"""Test-cache integrity and row lineage, using temporary historical artifacts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from backend.src.final_evaluation import ordered_row_ids
try:
    from backend.run_final_method_comparison import load_verified_cache
except ModuleNotFoundError as error:
    if error.name!='backend.run_final_method_comparison': raise
    load_verified_cache=None


class FinalMethodCacheTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(load_verified_cache,'Read-only Test cache loader not implemented')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.data={'texts':['một','hai','ba'],'labels':np.array([0,1,2]),
                   'sources':['hieu_sa','neu_esc','vsfc'],'sha256':'frozen-csv'}
        self.expected={'selected_alpha':.8,'selected_method':'ensemble','selection_dataset':'validation',
                       'selection_sha256':'frozen-selection','label_mapping':{'negative':0,'neutral':1,'positive':2},
                       'probability_columns':['negative','neutral','positive'],'step9_fingerprints':{'model':'unchanged'}}
        self.config={**self.expected,'test_sha256':self.data['sha256'],'sample_count':3,
                     'ordered_row_ids_sha256':hashlib.sha256('\n'.join(ordered_row_ids(self.data)).encode()).hexdigest()}
        self.arrays={'svm_probs':np.eye(3),'transformer_probs':np.eye(3),'y_true':self.data['labels'].copy(),
                     'sources':np.array(self.data['sources']),'row_ids':np.array(ordered_row_ids(self.data)),
                     'manifest_json':np.array(json.dumps(self.config))}
        self.write()

    def write(self):
        f=self.root/'evaluation_freeze.json';c=self.root/'test_probabilities.npz';r=self.root/'inference_record.json'
        f.write_text(json.dumps({'selected_before_test_inference':True,'configuration':self.config}),encoding='utf-8')
        np.savez_compressed(c,**self.arrays)
        record={'cache_sha256':hashlib.sha256(c.read_bytes()).hexdigest(),'freeze_sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
        r.write_text(json.dumps(record),encoding='utf-8')
        self.baseline={'dataset':{'sha256':self.data['sha256']},'sample_count':3,'selected_alpha':.8,
                       'label_order':['negative','neutral','positive'],'provenance':{'inference_record':record,
                       'evaluation_freeze_sha256':record['freeze_sha256']}}

    def load(self): return load_verified_cache(self.root,self.data,self.expected,self.baseline)

    def test_valid_historical_cache_is_read_without_writing_artifacts(self):
        before={p.name:p.read_bytes() for p in self.root.iterdir()}
        a,b=self.load();np.testing.assert_array_equal(a,np.eye(3));np.testing.assert_array_equal(b,np.eye(3))
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})

    def test_cache_integrity_hash_rejects_modified_probability_bytes(self):
        p=self.root/'test_probabilities.npz';p.write_bytes(p.read_bytes()+b'x')
        with self.assertRaises(ValueError): self.load()

    def test_ordered_text_hashes_reject_same_label_source_text_substitution(self):
        self.data['texts'][0]='thay đổi'
        with self.assertRaises(ValueError): self.load()

    def test_reordered_rows_truth_and_sources_are_rejected(self):
        for key in ('row_ids','y_true','sources'):
            original=self.arrays[key].copy();self.arrays[key]=original[::-1];self.write()
            with self.subTest(key=key),self.assertRaises(ValueError): self.load()
            self.arrays[key]=original

    def test_label_column_mapping_mismatch_is_rejected(self):
        self.config['probability_columns']=['positive','neutral','negative'];self.write()
        with self.assertRaises(ValueError): self.load()

    def test_changed_alpha_or_selection_dataset_is_rejected(self):
        for key,value in [('selected_alpha',.7),('selection_dataset','test')]:
            old=self.config[key];self.config[key]=value;self.write()
            with self.subTest(key=key),self.assertRaises(ValueError): self.load()
            self.config[key]=old

    def test_missing_cache_does_not_trigger_inference(self):
        (self.root/'test_probabilities.npz').unlink()
        with self.assertRaises(FileNotFoundError): self.load()

    def test_incomplete_or_inconsistent_manifest_is_rejected(self):
        self.arrays['manifest_json']=np.array('{}');self.write()
        with self.assertRaises(ValueError): self.load()

    def test_wrong_sample_count_and_invalid_probabilities_are_rejected(self):
        for value in (np.eye(3)[:2],np.ones((3,3))):
            self.arrays['svm_probs']=value;self.write()
            with self.subTest(shape=value.shape),self.assertRaises(ValueError): self.load()

    def test_frozen_artifact_fingerprint_mismatch_is_rejected(self):
        self.config['step9_fingerprints']={'model':'changed'};self.write()
        with self.assertRaises(ValueError): self.load()


if __name__=='__main__': unittest.main()

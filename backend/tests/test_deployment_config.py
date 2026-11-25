import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from backend.src.alpha_tuning import TIE_BREAK_RULE
from backend.src.config import LABELS,LABEL_TO_ID
from backend.src.deployment_config import build_deployment_config,load_deployment_config,validate_deployment_config,write_deployment_config


class DeploymentConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        (self.root/'models').mkdir(); (self.root/'models'/'model.bin').write_bytes(b'model')
        self.selection=self.root/'selection.json'; self.scores=self.root/'scores.csv'; self.freeze=self.root/'freeze.json'
        self.raw={'selection_dataset':'validation','test_dataset_used':False,'primary_metric':'macro_f1',
                  'selected_alpha':.8,'selected_validation_macro_f1':.83,'ensemble_validation_macro_f1':.83,
                  'svm_validation_macro_f1':.82,'transformer_validation_macro_f1':.79,
                  'probability_columns':LABELS,'label_mapping':LABEL_TO_ID,'alpha_grid':[0.,.8,1.],
                  'tied_alphas':[.8],'tie_break_rule':TIE_BREAK_RULE,
                  'fingerprints':{'models/model.bin':hashlib.sha256(b'model').hexdigest()}}
        self.rows=[{'alpha':0.,'macro_f1':.79},{'alpha':.8,'macro_f1':.83},{'alpha':1.,'macro_f1':.82}]
        self.save()

    def tearDown(self): self.temp.cleanup()

    def save(self):
        self.selection.write_text(json.dumps(self.raw),encoding='utf-8')
        with self.scores.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['alpha','macro_f1']); w.writeheader(); w.writerows(self.rows)

    def build(self):
        return build_deployment_config(self.selection,self.scores,self.freeze,backend_root=self.root)

    def test_reads_artifact_and_no_test_selection(self):
        config=self.build()
        self.assertEqual(config['selected_method'],'ensemble'); self.assertEqual(config['selected_alpha'],.8)
        self.assertEqual(config['selection_source'],'validation')
        self.assertEqual(config['default_view'],'user'); self.assertTrue(config['comparison_view_available'])
        self.assertEqual(config['document_aggregation_default'],'mean_probability')
        self.assertFalse(config['aggregation_validated_optimal'])

    def test_artifact_alpha_not_hardcoded(self):
        self.raw.update(selected_alpha=.7,alpha_grid=[0.,.7,1.],tied_alphas=[.7]); self.rows[1]['alpha']=.7; self.save()
        self.assertEqual(self.build()['selected_alpha'],.7)

    def test_invalid_alpha(self):
        for alpha in (True,-.1,1.1,float('nan'),float('inf'),'0.8'):
            self.raw['selected_alpha']=alpha; self.save()
            with self.subTest(alpha=alpha),self.assertRaises((ValueError,TypeError)): self.build()

    def test_missing_artifact_no_fallback(self):
        self.selection.unlink()
        with self.assertRaises(FileNotFoundError): self.build()

    def test_bad_mapping_source_metric(self):
        for key,value in [('selection_dataset','test'),('test_dataset_used',True),('primary_metric','accuracy'),
                          ('probability_columns',list(reversed(LABELS))),('label_mapping',{})]:
            original=self.raw[key]; self.raw[key]=value; self.save()
            with self.subTest(key=key),self.assertRaises(ValueError): self.build()
            self.raw[key]=original

    def test_mismatched_scores_grid_ties_and_rule(self):
        for key,value in [('selected_validation_macro_f1',.5),('tied_alphas',[.7]),('alpha_grid',[0.,1.]),('tie_break_rule','unknown')]:
            old=self.raw[key]; self.raw[key]=value; self.save()
            with self.subTest(key=key),self.assertRaises(ValueError): self.build()
            self.raw[key]=old

    def test_model_hash_mismatch(self):
        (self.root/'models'/'model.bin').write_bytes(b'changed')
        with self.assertRaises(ValueError): self.build()

    def test_endpoint_tie_rule_prefers_transformer_then_svm(self):
        self.raw.update(selected_alpha=0.,selected_validation_macro_f1=.82,ensemble_validation_macro_f1=.82,
                        transformer_validation_macro_f1=.82,tied_alphas=[0.,.8,1.])
        for r in self.rows: r['macro_f1']=.82
        self.save(); self.assertEqual(self.build()['selected_method'],'transformer')
        self.rows[0]['macro_f1']=.79
        self.raw.update(selected_alpha=1.,transformer_validation_macro_f1=.79,tied_alphas=[.8,1.])
        self.save(); self.assertEqual(self.build()['selected_method'],'svm')

    def test_frozen_selection_agrees_or_rejects(self):
        c=self.build()
        frozen={'selected_before_test_inference':True,'configuration':{
            'selected_alpha':c['selected_alpha'],'selected_method':c['selected_method'],
            'selection_sha256':c['artifact_hashes']['alpha_selection.json'],
            'validation_scores':c['validation_macro_f1'],'step9_fingerprints':self.raw['fingerprints'],
            'selection_dataset':'validation','label_mapping':LABEL_TO_ID,'probability_columns':LABELS}}
        self.freeze.write_text(json.dumps(frozen),encoding='utf-8')
        self.assertTrue(self.build()['frozen_selection_verified'])
        frozen['configuration']['selected_method']='svm'; self.freeze.write_text(json.dumps(frozen),encoding='utf-8')
        with self.assertRaises(ValueError): self.build()

    def test_frozen_dataset_and_label_contract_must_match(self):
        c=self.build()
        frozen={'selected_before_test_inference':True,'configuration':{
            'selected_alpha':c['selected_alpha'],'selected_method':c['selected_method'],
            'selection_sha256':c['artifact_hashes']['alpha_selection.json'],
            'validation_scores':c['validation_macro_f1'],'step9_fingerprints':self.raw['fingerprints'],
            'selection_dataset':'validation','label_mapping':LABEL_TO_ID,'probability_columns':LABELS}}
        for key,value in [('selection_dataset','test'),('label_mapping',{'negative':2,'neutral':1,'positive':0}),
                          ('probability_columns',list(reversed(LABELS)))]:
            bad=json.loads(json.dumps(frozen)); bad['configuration'][key]=value
            self.freeze.write_text(json.dumps(bad))
            with self.subTest(key=key),self.assertRaises(ValueError): self.build()

    def test_invalid_deployment_method_alpha_and_aggregation(self):
        c=self.build()
        for key,value in [('selected_method','unknown'),('selected_alpha',False),('default_view','research'),
                          ('aggregation_validated_optimal',True),('document_aggregation_default','majority_vote')]:
            changed={**c,key:value}
            with self.subTest(key=key),self.assertRaises((ValueError,TypeError)): validate_deployment_config(changed)

    def test_write_load_and_stale_config_rejected(self):
        p=self.root/'deployment.json'; c=self.build()
        write_deployment_config(p,c)
        args={'selection_path':self.selection,'scores_path':self.scores,'freeze_path':self.freeze,'backend_root':self.root}
        self.assertEqual(load_deployment_config(p,**args),c)
        c['selected_alpha']=.5; p.write_text(json.dumps(c))
        with self.assertRaises(ValueError): load_deployment_config(p,**args)

    def test_test_scores_are_display_only_even_when_test_favors_svm(self):
        p=self.root/'outputs'/'final_evaluation'/'test_metrics.json'; p.parent.mkdir(parents=True)
        p.write_text(json.dumps({'selected_method':'ensemble','selected_alpha':.8,'selection_dataset':'validation',
                                'models':{'svm':{'macro_f1':.99},'transformer':{'macro_f1':.9},'ensemble':{'macro_f1':.1}}}))
        c=self.build()
        self.assertEqual(c['selected_method'],'ensemble'); self.assertEqual(c['selected_alpha'],.8)
        self.assertEqual(c['test_macro_f1']['svm'],.99)

    def test_missing_scores_and_invalid_endpoint_stop(self):
        self.scores.unlink()
        with self.assertRaises(FileNotFoundError): self.build()
        self.rows[-1]['macro_f1']=.4; self.save()
        with self.assertRaises(ValueError): self.build()

    def test_no_dataset_file_is_opened_for_deployment_verification(self):
        self.raw['fingerprints']['data/test.csv']='recorded-only'; self.save()
        self.assertEqual(self.build()['selected_method'],'ensemble')


if __name__=='__main__': unittest.main()

"""Current demo must use the validated model version and matching metadata."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
import shutil
from unittest.mock import patch

from backend.src.deployment_config import build_deployment_config, load_deployment_config
from backend.tests import test_deployment_config as legacy_tests


class ExplicitProfileMetadataTests(unittest.TestCase):
    setUp=legacy_tests.DeploymentConfigTests.setUp
    tearDown=legacy_tests.DeploymentConfigTests.tearDown
    save=legacy_tests.DeploymentConfigTests.save
    def test_explicit_test_report_avoids_legacy_metrics(self):
        self.assertIn('test_path', inspect.signature(build_deployment_config).parameters,
                      'Versioned deployment must not read legacy Test scores')
        test_path=self.root/'epoch21_test.json'
        test_path.write_text(json.dumps({'selected_method':'ensemble','selected_alpha':.8,
            'selection_dataset':'validation','models':{key:{'macro_f1':value} for key,value in
            [('svm',.81),('transformer',.80),('ensemble',.84)]}}),encoding='utf-8')
        cfg=build_deployment_config(self.selection,self.scores,self.freeze,
                                    backend_root=self.root,test_path=test_path)
        self.assertEqual(cfg['test_macro_f1'],{'svm':.81,'transformer':.80,'ensemble':.84})


class ActiveEpoch21Tests(unittest.TestCase):
    def test_active_demo_uses_epoch21_and_validation_selected_weights(self):
        cfg=load_deployment_config()
        self.assertEqual(cfg['selected_alpha'],.7,'Demo still uses the old 80/20 selection')
        self.assertEqual(cfg['selected_method'],'ensemble')
        self.assertEqual(cfg['artifact_profile'],'transformer_epoch21')
        self.assertEqual(cfg['transformer_checkpoint_epoch'],21)
        self.assertAlmostEqual(cfg['validation_macro_f1']['ensemble'],.8249764545971283)
        self.assertAlmostEqual(cfg['test_macro_f1']['transformer'],.804909705502149)
        self.assertAlmostEqual(cfg['test_macro_f1']['ensemble'],.8252706095647272)
        self.assertTrue(cfg['frozen_selection_verified'])

    def test_modified_deployment_lock_is_rejected(self):
        from backend.src import epoch21_deployment as current
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for name in ['alpha_selection.json','alpha_scores.csv','test_metrics.json','deployment_lock.json']:
                shutil.copyfile(current.ARTIFACT_DIR/name,folder/name)
            lock=json.loads((folder/'deployment_lock.json').read_text(encoding='utf-8'))
            lock['selected_alpha']=.8
            (folder/'deployment_lock.json').write_text(json.dumps(lock),encoding='utf-8')
            with patch.object(current,'ARTIFACT_DIR',folder),self.assertRaisesRegex(ValueError,'lock'):
                current.build_epoch21_deployment_config()

    def test_profile_startup_does_not_read_validation_or_test_csv(self):
        original=Path.open
        def audited(path,*args,**kwargs):
            self.assertNotIn(Path(path).name,['val.csv','test.csv','train.csv'])
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',audited):
            self.assertEqual(load_deployment_config()['selected_alpha'],.7)


if __name__=='__main__': unittest.main()

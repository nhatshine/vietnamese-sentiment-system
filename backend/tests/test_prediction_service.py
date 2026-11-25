import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from backend.src.prediction_service import PredictionService,present_result


def config():
    return {'selection_source':'validation','selection_metric':'macro_f1','selected_method':'ensemble','selected_alpha':.8,
            'default_view':'user','comparison_view_available':True,'document_aggregation_default':'mean_probability',
            'aggregation_validated_optimal':False,'validation_macro_f1':{'svm':.82,'transformer':.79,'ensemble':.83},
            'test_macro_f1':{'svm':.81,'transformer':.78,'ensemble':.82},'metric_scope':'dataset_not_input_confidence',
            'frozen_selection_verified':True}


class PredictionServiceTests(unittest.TestCase):
    def setUp(self):
        self.calls={'svm':[],'transformer':[]}
        def svm(text):
            self.calls['svm'].append(text)
            return {'negative':.9,'neutral':.05,'positive':.05}
        def transformer(text):
            self.calls['transformer'].append(text)
            return {'negative':.1,'neutral':.3,'positive':.6}
        self.cfg=config()
        with patch('backend.src.prediction_service.load_deployment_config',return_value=self.cfg):
            self.service=PredictionService(svm_predictor=svm,transformer_predictor=transformer)

    def test_single_raw_text_final_and_full_comparison(self):
        text='  Sản phẩm rất tốt!  '
        r=self.service.analyze(text)
        self.assertEqual(r['input_text'],text); self.assertEqual(r['sentence_count'],1)
        self.assertEqual(set(r['comparison']),{'svm','transformer','ensemble'})
        self.assertEqual(r['final_result'],r['comparison']['ensemble'])
        self.assertAlmostEqual(r['final_result']['probabilities']['negative'],.74)
        self.assertEqual(r['final_result']['method'],'ensemble')
        self.assertNotEqual(r['final_result']['confidence'],r['comparison']['svm']['confidence'])
        self.assertEqual(self.calls['svm'],['Sản phẩm rất tốt!'])
        self.assertEqual(self.calls['transformer'],['Sản phẩm rất tốt!'])

    def test_multiple_sentences_keep_order_and_no_duplicate_inference(self):
        r=self.service.analyze('Camera đẹp.\nPin yếu! Giao nhanh?')
        expected=['Camera đẹp.','Pin yếu!','Giao nhanh?']
        self.assertEqual([s['sentence'] for s in r['sentences']],expected)
        self.assertEqual([s['sentence_index'] for s in r['sentences']],[0,1,2])
        self.assertEqual(self.calls,{'svm':expected,'transformer':expected})
        for s in r['sentences']: self.assertEqual(s['final_result'],s['comparison']['ensemble'])
        self.assertEqual(r['aggregation_info']['method'],'mean_probability')
        self.assertFalse(r['aggregation_info']['validated_optimal'])
        self.assertEqual(r['final_result'],r['comparison']['ensemble'])

    def test_mean_probability_not_last_sentence_or_majority_distribution(self):
        def svm(text):
            return {'negative':.9,'neutral':.05,'positive':.05} if text=='Xấu.' else {'negative':.05,'neutral':.05,'positive':.9}
        def transformer(text): return {'negative':.05,'neutral':.9,'positive':.05}
        with patch('backend.src.prediction_service.load_deployment_config',return_value=config()):
            service=PredictionService(svm_predictor=svm,transformer_predictor=transformer)
        r=service.analyze('Đẹp. Tốt. Xấu.')
        self.assertEqual(r['final_result']['label'],'positive')
        self.assertEqual(r['sentences'][-1]['final_result']['label'],'negative')
        self.assertAlmostEqual(r['final_result']['probabilities']['positive'],1.51/3)
        self.assertNotEqual(r['final_result']['confidence'],2/3)
        for m in ('svm','transformer','ensemble'):
            self.assertTrue({'majority_vote','mean_probability','confidence_weighted'}.issubset(r['document_results'][m]))
            self.assertEqual(r['final_result']['probabilities'],r['document_results']['ensemble']['mean_probability']['probabilities'])

    def test_selected_svm_or_transformer_when_explicit_config_selects_them(self):
        for m in ('svm','transformer'):
            c=config(); c['selected_method']=m
            with patch('backend.src.prediction_service.load_deployment_config',return_value=c):
                s=PredictionService(svm_predictor=self.service._svm_predictor,transformer_predictor=self.service._transformer_predictor)
            self.assertEqual(s.analyze('Văn bản.')['final_result']['method'],m)

    def test_invalid_input_before_predictors(self):
        for text in (None,1,[],{}):
            with self.subTest(text=text),self.assertRaises(TypeError): self.service.analyze(text)
        for text in ('',' \n\t '):
            with self.subTest(text=text),self.assertRaises(ValueError): self.service.analyze(text)
        self.assertEqual(self.calls,{'svm':[],'transformer':[]})

    def test_unicode_newlines_and_long_text_retained(self):
        text='  Tiếng Việt ĐẶC BIỆT!\n'+('tuyệt vời '*300)+'  '
        r=self.service.analyze(text)
        self.assertEqual(r['input_text'],text)
        self.assertEqual(len(r['sentences']),2)
        self.assertEqual(self.calls['svm'][1],('tuyệt vời '*300).strip())

    def test_user_and_research_presentation_use_same_inference(self):
        r=self.service.analyze('Xin chào. Tốt!'); before=copy.deepcopy(r)
        user=present_result(r,'user'); research=present_result(r,'research')
        self.assertEqual(user['final_result'],research['final_result'])
        self.assertNotIn('comparison',user); self.assertNotIn('document_results',user); self.assertNotIn('research_info',user)
        self.assertNotIn('comparison',user['sentences'][0])
        self.assertEqual(set(research['comparison']),{'svm','transformer','ensemble'})
        self.assertEqual(research['research_info']['metric_scope'],'dataset_not_input_confidence')
        self.assertEqual(r,before)
        self.assertEqual(len(self.calls['svm']),2); self.assertEqual(len(self.calls['transformer']),2)

    def test_presented_mutations_do_not_affect_canonical_response(self):
        r=self.service.analyze('Xin chào!'); before=copy.deepcopy(r)
        view=present_result(r,'research'); view['comparison']['svm']['probabilities']['negative']=0
        self.assertEqual(r,before)

    def test_invalid_view(self):
        r=self.service.analyze('Xin chào.')
        for view in ('other',None,True):
            with self.subTest(view=view),self.assertRaises(ValueError): present_result(r,view)

    def test_config_caller_cannot_change_frozen_alpha(self):
        self.cfg['selected_alpha']=0
        self.service.config['selected_alpha']=0
        r=self.service.analyze('Xin chào.')
        self.assertEqual(r['selection_info']['selected_alpha'],.8)
        self.assertAlmostEqual(r['final_result']['probabilities']['negative'],.74)

    def test_json_serializable_and_dataset_metrics_separate(self):
        r=self.service.analyze('Xin chào. Tốt!')
        self.assertEqual(json.loads(json.dumps(r,ensure_ascii=False,allow_nan=False)),r)
        self.assertNotIn('macro_f1',r['final_result'])
        self.assertEqual(r['research_info']['validation_macro_f1']['ensemble'],.83)

    def test_research_reports_explicit_frozen_ensemble_weights(self):
        r=self.service.analyze('Xin chào. Tốt!')
        weights=present_result(r,'research')['research_info']['ensemble_weights']
        self.assertEqual(weights['svm'],.8)
        self.assertAlmostEqual(weights['transformer'],.2)
        self.assertNotIn('research_info',present_result(r,'user'))

    def test_error_on_any_sentence_is_not_silently_skipped(self):
        def bad(text):
            if text=='!!!': raise ValueError('rỗng sau preprocessing')
            return {'negative':.9,'neutral':.05,'positive':.05}
        with patch('backend.src.prediction_service.load_deployment_config',return_value=config()):
            service=PredictionService(svm_predictor=bad,transformer_predictor=self.service._transformer_predictor)
        with self.assertRaisesRegex(ValueError,'svm.*!!!'): service.analyze('Tốt. !!! Sau.')

    def test_invalid_predictor_and_malformed_unselected_prediction(self):
        with patch('backend.src.prediction_service.load_deployment_config',return_value=config()):
            with self.assertRaises(TypeError): PredictionService(svm_predictor=42)
            s=PredictionService(svm_predictor=lambda t:{'negative':1},transformer_predictor=self.service._transformer_predictor)
        with self.assertRaises(ValueError): s.analyze('Xin chào.')

    def test_production_defaults_to_real_predictors(self):
        from backend.src import real_svm,real_transformer
        with patch('backend.src.prediction_service.load_deployment_config',return_value=config()): service=PredictionService()
        self.assertIs(service._svm_predictor,real_svm.predict_svm)
        self.assertIs(service._transformer_predictor,real_transformer.predict_transformer)

    def test_service_inference_never_opens_test_dataset(self):
        original=Path.open
        def guard(path,*args,**kwargs):
            if path.name=='test.csv': raise AssertionError('Test dataset forbidden during inference')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guard): r=self.service.analyze('Xin chào.')
        self.assertEqual(r['sentence_count'],1)


class RealServiceIntegrationTests(unittest.TestCase):
    def test_real_models_cached_and_each_preprocess_called_once_per_sentence(self):
        from backend.src import real_svm,real_transformer
        from backend.models.transformer import tokenizer
        real_svm.load_model.cache_clear(); real_transformer._load_cached_model.cache_clear()
        with patch('backend.src.prediction_service.load_deployment_config',return_value=config()), \
             patch.object(real_svm,'clean_text',wraps=real_svm.clean_text) as svm_clean, \
             patch.object(tokenizer,'clean_text',wraps=tokenizer.clean_text) as transformer_clean:
            service=PredictionService()
            one=service.analyze('Sản phẩm này rất tốt.')
            two=service.analyze('Sản phẩm này bình thường.')
            self.assertEqual(svm_clean.call_count,2); self.assertEqual(transformer_clean.call_count,2)
            self.assertEqual(real_svm.load_model.cache_info().misses,1)
            self.assertEqual(real_transformer._load_cached_model.cache_info().misses,1)
            self.assertEqual(one['final_result'],one['comparison']['ensemble'])
            self.assertEqual(two['final_result'],two['comparison']['ensemble'])
            json.dumps(two,allow_nan=False)

    def test_real_cleaned_empty_sentence_fails_clearly(self):
        with patch('backend.src.prediction_service.load_deployment_config',return_value=config()): service=PredictionService()
        with self.assertRaisesRegex(ValueError,'svm.*!!!'): service.analyze('!!!')


class DemoPresentationTests(unittest.TestCase):
    setUp=PredictionServiceTests.setUp
    def test_demo_default_hides_model_comparison(self):
        from backend.demo_final_pipeline import render_result
        r=self.service.analyze('Xin chào.')
        user=render_result(r,'user'); research=render_result(r,'research')
        self.assertIn('FINAL RESULT',user)
        self.assertNotIn('SVM RESULT',user); self.assertNotIn('TRANSFORMER RESULT',user)
        self.assertIn('SVM RESULT',research); self.assertIn('TRANSFORMER RESULT',research); self.assertIn('ENSEMBLE RESULT',research)
        self.assertEqual(len(self.calls['svm']),1); self.assertEqual(len(self.calls['transformer']),1)

    def test_demo_both_views_infer_only_six_inputs(self):
        import contextlib,io
        from backend.demo_final_pipeline import main
        service=self.service
        with patch('backend.demo_final_pipeline.PredictionService',return_value=service),contextlib.redirect_stdout(io.StringIO()) as output:
            main(['--both'])
        # Five standalone sentences plus three sentences in the paragraph.
        self.assertEqual(len(self.calls['svm']),8); self.assertEqual(len(self.calls['transformer']),8)
        self.assertEqual(output.getvalue().count('SVM RESULT'),6)


if __name__=='__main__': unittest.main()

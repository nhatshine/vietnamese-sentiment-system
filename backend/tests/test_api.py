"""HTTP tests with real service logic and fake predictor boundaries."""
import copy
import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend.src.api import create_app
from backend.src.prediction_service import PredictionService
from backend.tests.test_prediction_service import config


class FakeService:
    def __init__(self):
        self.calls=[]; self.predictor_calls={'svm':0,'transformer':0}; self.config_reads=0
        def svm(text):
            self.predictor_calls['svm']+=1
            return {'negative':.9,'neutral':.05,'positive':.05}
        def transformer(text):
            self.predictor_calls['transformer']+=1
            return {'negative':.1,'neutral':.3,'positive':.6}
        with patch('backend.src.prediction_service.load_deployment_config',return_value=config()):
            self.pipeline=PredictionService(svm_predictor=svm,transformer_predictor=transformer)

    @property
    def config(self):
        self.config_reads+=1
        return self.pipeline.config

    def analyze(self,text):
        self.calls.append(text)
        return self.pipeline.analyze(text)


class APITests(unittest.TestCase):
    def setUp(self):
        self.service=FakeService(); self.factory_calls=0
        def factory():
            self.factory_calls+=1; return self.service
        self.app=create_app(service_factory=factory)
        self.client=TestClient(self.app,raise_server_exceptions=False)
        self.client.__enter__()

    def tearDown(self): self.client.__exit__(None,None,None)

    def test_singleton_and_metadata_do_not_infer(self):
        for _ in range(3):
            self.assertEqual(self.client.get('/api/health').status_code,200)
            self.assertEqual(self.client.get('/api/model-info').status_code,200)
        self.assertEqual(self.factory_calls,1); self.assertEqual(self.service.calls,[])
        self.assertEqual(self.service.config_reads,1)

    def test_single_sentence_raw_full_precision_selected_not_max_confidence(self):
        text='  Tiếng Việt rất tốt!  '
        response=self.client.post('/api/analyze',json={'text':text})
        self.assertEqual(response.status_code,200,response.text)
        r=response.json()
        self.assertEqual(r['input_text'],text); self.assertEqual(r['sentence_count'],1)
        self.assertEqual(r['final_result'],r['comparison']['ensemble'])
        self.assertEqual(r['selection_info']['selected_alpha'],.8)
        self.assertAlmostEqual(r['final_result']['confidence'],.74)
        self.assertNotEqual(r['final_result']['confidence'],r['comparison']['svm']['confidence'])
        self.assertEqual(self.service.calls,[text])
        self.assertEqual(self.service.predictor_calls,{'svm':1,'transformer':1})
        json.dumps(r,allow_nan=False)

    def test_multiple_sentences_and_all_research_data(self):
        r=self.client.post('/api/analyze',json={'text':'Đẹp.\nPin yếu! Giao nhanh?'}).json()
        self.assertEqual(r['sentence_count'],3)
        self.assertEqual([s['sentence'] for s in r['sentences']],['Đẹp.','Pin yếu!','Giao nhanh?'])
        self.assertEqual(set(r['comparison']),{'svm','transformer','ensemble'})
        self.assertEqual(set(r['document_results']),{'svm','transformer','ensemble'})
        self.assertEqual(r['aggregation_info']['method'],'mean_probability')
        self.assertFalse(r['aggregation_info']['validated_optimal'])
        self.assertEqual(r['display_info']['default_view'],'user')
        self.assertEqual(r['research_info']['metric_scope'],'dataset_not_input_confidence')
        self.assertEqual(self.service.predictor_calls,{'svm':3,'transformer':3})

    def test_repeated_requests_keep_singleton_and_frozen_config(self):
        before=copy.deepcopy(self.service.pipeline.config)
        for _ in range(2): self.assertEqual(self.client.post('/api/analyze',json={'text':'Tốt.'}).status_code,200)
        self.assertEqual(self.factory_calls,1); self.assertEqual(before,self.service.pipeline.config)

    def test_invalid_text_schema_and_overrides_422_without_inference(self):
        bodies=[{}, {'text':''},{'text':' \n\t '},{'text':None},{'text':42},{'text':True},{'text':[]},
                {'text':{}},{'text':'x','alpha':.2},{'text':'x','selected_method':'svm'},
                {'text':'x','model_path':'C:/secret'},{'text':'x','view':'research'}]
        for body in bodies:
            with self.subTest(body=body): self.assertEqual(self.client.post('/api/analyze',json=body).status_code,422)
        self.assertEqual(self.service.calls,[])

    def test_bad_json_and_non_object_422(self):
        for body in ('{invalid','[]','null','"text"'):
            with self.subTest(body=body):
                r=self.client.post('/api/analyze',content=body,headers={'Content-Type':'application/json'})
                self.assertEqual(r.status_code,422)
        self.assertEqual(self.service.calls,[])

    def test_character_limit_rejects_without_truncating(self):
        with TestClient(create_app(service_factory=lambda:self.service,max_text_chars=4)) as client:
            self.assertEqual(client.post('/api/analyze',json={'text':'12345'}).status_code,422)
            r=client.post('/api/analyze',json={'text':'1234'})
            self.assertEqual(r.status_code,200); self.assertEqual(r.json()['input_text'],'1234')

    def test_unicode_surrogate_422(self):
        r=self.client.post('/api/analyze',content=b'{"text":"\\ud800"}',headers={'Content-Type':'application/json'})
        self.assertEqual(r.status_code,422)

    def test_invalid_utf8_transport_is_safe422(self):
        r=self.client.post('/api/analyze',content=b'{"text":"\xff"}',headers={'Content-Type':'application/json'})
        self.assertEqual(r.status_code,422)
        self.assertNotIn('final_result',r.json())

    def test_inference_failure_safe500_never_fake_prediction(self):
        for error in (RuntimeError('C:/secret/checkpoint.pth private raw'),ValueError('invalid model probability C:/secret')):
            with self.subTest(error=type(error).__name__),patch.object(self.service,'analyze',side_effect=error):
                r=self.client.post('/api/analyze',json={'text':'secret input'})
                self.assertEqual(r.status_code,500)
                self.assertNotIn('final_result',r.json()); self.assertNotIn('C:/secret',r.text); self.assertNotIn('secret input',r.text)

    def test_cleaned_empty_safe422(self):
        inner=ValueError('Văn bản rỗng sau preprocessing; không thực hiện inference.')
        outer=ValueError('svm: không phân tích được câu !!!'); outer.__cause__=inner
        with patch.object(self.service,'analyze',side_effect=outer): r=self.client.post('/api/analyze',json={'text':'!!!'})
        self.assertEqual(r.status_code,422); self.assertNotIn('final_result',r.json())

    def test_not_ready_503_and_liveness_still_ok(self):
        self.app.state.ready=False
        self.assertEqual(self.client.post('/api/analyze',json={'text':'Tốt.'}).status_code,503)
        h=self.client.get('/api/health').json()
        self.assertEqual(h['status'],'ok'); self.assertFalse(h['model_ready'])

    def test_health_is_safe_and_truthful(self):
        r=self.client.get('/api/health')
        self.assertEqual(r.json(),{'status':'ok','service':'sentiment-analysis-api','version':'1.0.0','model_ready':True})
        self.assertNotIn('C:',r.text)

    def test_model_info_whitelist_and_real_config_scores(self):
        r=self.client.get('/api/model-info'); d=r.json()
        self.assertEqual(r.status_code,200)
        self.assertEqual(d['selected_method'],'ensemble'); self.assertEqual(d['selected_alpha'],.8)
        self.assertEqual(d['validation_metrics']['ensemble']['macro_f1'],.83)
        self.assertEqual(d['test_metrics']['svm']['macro_f1'],.81)
        self.assertEqual(d['selection_source'],'validation')
        self.assertFalse(d['document_aggregation']['validated_optimal'])
        self.assertNotIn('fingerprints',r.text); self.assertNotIn('C:',r.text)

    def test_nonfinite_or_non_json_output_safe500(self):
        for value in (float('nan'),float('inf'),object()):
            r=self.service.pipeline.analyze('Tốt.'); r['research_info']['bad_extra']=value
            with self.subTest(value=type(value).__name__),patch.object(self.service,'analyze',return_value=r):
                output=self.client.post('/api/analyze',json={'text':'Tốt.'})
                self.assertEqual(output.status_code,500); self.assertNotIn('NaN',output.text)

    def test_wrong_final_or_frozen_selection_safe500(self):
        for field in ('final','alpha'):
            r=self.service.pipeline.analyze('Tốt.')
            if field=='final': r['final_result']=r['comparison']['svm']
            else: r['selection_info']['selected_alpha']=.5
            with self.subTest(field=field),patch.object(self.service,'analyze',return_value=r):
                self.assertEqual(self.client.post('/api/analyze',json={'text':'Tốt.'}).status_code,500)

    def test_invalid_unicode_output_has_safe_json500(self):
        data=self.service.pipeline.analyze('Tốt.'); data['research_info']['bad_unicode']='\ud800'
        with patch.object(self.service,'analyze',return_value=data): r=self.client.post('/api/analyze',json={'text':'Tốt.'})
        self.assertEqual(r.status_code,500)
        self.assertTrue(r.headers['content-type'].startswith('application/json'))
        self.assertIn('detail',r.json())

    def test_canonical_response_unfiltered_unrounded(self):
        expected=self.service.pipeline.analyze('Tốt.'); expected['research_info']['extra']={'full_precision':.12345678912345678}
        with patch.object(self.service,'analyze',return_value=expected): r=self.client.post('/api/analyze',json={'text':'Tốt.'})
        self.assertEqual(r.json(),expected)

    def test_cors_local_origins_and_preflight(self):
        for origin in ('http://localhost:5173','http://127.0.0.1:5173'):
            with self.subTest(origin=origin):
                r=self.client.options('/api/analyze',headers={'Origin':origin,'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
                self.assertEqual(r.status_code,200); self.assertEqual(r.headers['access-control-allow-origin'],origin)
                self.assertNotIn('access-control-allow-credentials',r.headers)

    def test_unknown_cors_origin_not_granted(self):
        r=self.client.get('/api/health',headers={'Origin':'https://unknown.example'})
        self.assertNotIn('access-control-allow-origin',r.headers)

    def test_swagger_openapi(self):
        self.assertEqual(self.client.get('/docs').status_code,200)
        schema=self.client.get('/openapi.json').json()
        self.assertEqual(set(schema['paths']),{'/api/analyze','/api/health','/api/model-info'})
        self.assertEqual(schema['info']['version'],'1.0.0')
        self.assertEqual(schema['components']['schemas']['AnalyzeRequest']['additionalProperties'],False)

    def test_concurrent_analyze_serialized(self):
        active=0; maximum=0; guard=threading.Lock(); original=self.service.analyze
        def slow(text):
            nonlocal active,maximum
            with guard: active+=1; maximum=max(maximum,active)
            time.sleep(.03)
            try: return original(text)
            finally:
                with guard: active-=1
        with patch.object(self.service,'analyze',side_effect=slow),ThreadPoolExecutor(max_workers=3) as pool:
            responses=list(pool.map(lambda _:self.client.post('/api/analyze',json={'text':'Tốt.'}),range(3)))
        self.assertEqual([r.status_code for r in responses],[200]*3); self.assertEqual(maximum,1)


class APIStartupTests(unittest.TestCase):
    def test_startup_failure_no_mock_fallback(self):
        def bad(): raise ValueError('C:/secret missing model')
        with self.assertRaisesRegex(RuntimeError,'khởi tạo'):
            with TestClient(create_app(service_factory=bad)): pass

    def test_liveness_before_lifespan_does_not_claim_ready(self):
        app=create_app(); client=TestClient(app)
        self.assertFalse(client.get('/api/health').json()['model_ready'])
        self.assertEqual(client.get('/api/model-info').status_code,503)

    def test_environment_settings(self):
        with patch.dict('os.environ',{'SENTIMENT_MAX_TEXT_CHARS':'7','SENTIMENT_CORS_ORIGINS':'http://localhost:3000'}): app=create_app(service_factory=FakeService)
        with TestClient(app) as client:
            self.assertEqual(client.post('/api/analyze',json={'text':'12345678'}).status_code,422)
            r=client.get('/api/health',headers={'Origin':'http://localhost:3000'})
            self.assertEqual(r.headers['access-control-allow-origin'],'http://localhost:3000')

    def test_invalid_runtime_settings(self):
        for limit in (0,-1,True,'x'):
            with self.subTest(limit=limit),self.assertRaises((TypeError,ValueError)): create_app(max_text_chars=limit)
        with self.assertRaises(ValueError): create_app(cors_origins=['*'])


if __name__=='__main__': unittest.main()

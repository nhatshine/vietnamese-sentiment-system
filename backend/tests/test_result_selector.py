import copy
import unittest

from backend.src.result_selector import select_result


def predictions():
    return {
        'svm':{'method':'svm','label':'negative','confidence':.9,'probabilities':{'negative':.9,'neutral':.05,'positive':.05}},
        'transformer':{'method':'transformer','label':'positive','confidence':.6,'probabilities':{'negative':.1,'neutral':.3,'positive':.6}},
        'ensemble':{'method':'ensemble','label':'neutral','confidence':.5,'probabilities':{'negative':.3,'neutral':.5,'positive':.2},'weights':{'svm':.8,'transformer':.2}},
    }


class ResultSelectorTests(unittest.TestCase):
    def test_each_supported_method_returns_exact_prediction(self):
        p=predictions()
        for method in p:
            with self.subTest(method=method): self.assertEqual(select_result(p,method),p[method])

    def test_does_not_select_highest_confidence(self):
        self.assertEqual(select_result(predictions(),'ensemble')['label'],'neutral')

    def test_result_and_input_do_not_alias(self):
        p=predictions(); before=copy.deepcopy(p)
        out=select_result(p,'ensemble'); out['probabilities']['negative']=0; out['weights']['svm']=0
        self.assertEqual(p,before)

    def test_invalid_method_and_structure_fail(self):
        for method in ('unknown',None,True):
            with self.subTest(method=method),self.assertRaises((ValueError,TypeError)): select_result(predictions(),method)
        for value in (None,[],{}, {'svm':predictions()['svm']}):
            with self.subTest(value=value),self.assertRaises((ValueError,TypeError)): select_result(value,'svm')

    def test_invalid_unselected_model_is_rejected(self):
        for field,value in [('label','positive'),('confidence',.2),('confidence',float('nan')),
                            ('confidence',True),('method','ensemble'),('probabilities',{'negative':1})]:
            p=predictions(); p['svm'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises((ValueError,TypeError)): select_result(p,'ensemble')

    def test_missing_fields_and_invalid_probabilities_rejected(self):
        for key in ('label','confidence','probabilities'):
            p=predictions(); del p['transformer'][key]
            with self.subTest(key=key),self.assertRaises(ValueError): select_result(p,'ensemble')
        p=predictions(); p['transformer']['probabilities']['positive']=float('inf')
        with self.assertRaises(ValueError): select_result(p,'ensemble')


if __name__=='__main__': unittest.main()

"""Check the actual CPU checkpoint, raw examples and deployment formulas."""
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch

from backend.src.epoch21_deployment import ARTIFACT_DIR,load_epoch21_transformer,predict_transformer_epoch21
from backend.src.real_transformer import encode_cleaned_text
from backend.src.api_dependencies import create_prediction_service,build_model_info
from backend.src.api_schemas import AnalyzeResponse


def main():
    torch.set_num_threads(1)
    model=load_epoch21_transformer()
    proof={'device':str(model.device),'checkpoint_epoch':21,'split_checks':{}}
    archive=Path.home()/'Downloads/transformer_retrain_30ep-20261122T164834Z-1-001.zip'
    with ZipFile(archive) as z:
        for split in ['val','test']:
            rows=list(csv.DictReader(io.StringIO(z.read(f'transformer_retrain_30ep/metrics/{split}_predictions.csv').decode('utf-8-sig'))))
            rng=np.random.default_rng(42)
            indices=sorted(set([0,1,2,len(rows)-1,*rng.choice(len(rows),60,replace=False).tolist()]))
            encoded=[encode_cleaned_text(rows[i]['text'],model.vocab,model.max_len) for i in indices]
            with torch.inference_mode():
                ids=torch.tensor([x[0] for x in encoded],dtype=torch.long)
                masks=torch.tensor([x[1] for x in encoded],dtype=torch.long)
                actual=model.model(ids,masks).softmax(-1).cpu().numpy()
            saved=np.asarray([[float(rows[i]['p_'+label]) for label in ['negative','neutral','positive']] for i in indices])
            assert np.allclose(actual,saved,atol=1e-5,rtol=0)
            assert np.array_equal(actual.argmax(1),saved.argmax(1))
            proof['split_checks'][split]={'n_checked':len(indices),'max_probability_difference':float(np.max(np.abs(actual-saved))),'labels_match':True}
    examples=['Sản phẩm tuyệt vời, rất hài lòng','Dịch vụ quá tệ, không muốn quay lại','Tôi vừa nhận được hàng hôm nay']
    example_rows=[]
    for text in examples:
        probs=predict_transformer_epoch21(text)
        example_rows.append({'text':text,**{'p_'+k:v for k,v in probs.items()}})
    with (ARTIFACT_DIR/'bang9_inference_epoch21.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(example_rows[0]));w.writeheader();w.writerows(example_rows)
    service=create_prediction_service();info=build_model_info(service.config)
    proof['model_info']=info;proof['raw_examples']=example_rows
    inputs=['Sản phẩm này rất tốt.','Tôi rất thất vọng vì sản phẩm quá tệ.','Sản phẩm này bình thường.',
            'Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!',
            'Giá sản phẩm là 1.5 triệu đồng. TS. Nam đánh giá chất lượng tốt.']
    records=[]
    for text in inputs:
        response=service.analyze(text);AnalyzeResponse.model_validate(response)
        assert response['selection_info']['selected_alpha']==.7
        for sent in response['sentences']:
            p=sent['comparison']
            for label in ['negative','neutral','positive']:
                expected=.7*p['svm']['probabilities'][label]+.3*p['transformer']['probabilities'][label]
                assert abs(p['ensemble']['probabilities'][label]-expected)<1e-12
        records.append({'text':text,'response':response})
    proof['records']=records
    (ARTIFACT_DIR/'cpu_verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'checkpoint':21,'alpha':info['selected_alpha'],'checks':proof['split_checks'],'examples':example_rows,
                      'paragraph_result':records[3]['response']['final_result']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

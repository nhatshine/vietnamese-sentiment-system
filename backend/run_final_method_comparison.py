"""Step 14.2: read-only historical Test cache, fixed methods, separate outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import numpy as np

from .run_alpha_tuning import BACKEND, collect_fingerprints
from .run_max_confidence_comparison import markdown_table, prediction_rows, write_csv
from .src.alpha_tuning import compute_ensemble_probabilities
from .src.config import LABELS
from .src.final_evaluation import file_sha256, load_frozen_selection, load_test_dataset, ordered_row_ids
from .src.final_method_comparison import assert_ensemble_baseline, evaluate_fixed_methods, paired_statistics
from .src.max_confidence import FIXED_ALPHA, select_max_confidence

ROOT=BACKEND.parent
HISTORICAL=BACKEND/'outputs/final_evaluation'
OUTPUT_ROOT=BACKEND/'outputs/final_method_comparison'
REPORT_PATH=ROOT/'docs/FINAL_METHOD_COMPARISON.md'
STATISTICAL_PLAN={'mcnemar':'exact two-sided binomial','significance_level':.05,
                  'bootstrap_replicates':10000,'bootstrap_seed':42,'bootstrap_confidence_level':.95,
                  'bootstrap_sampling':'paired IID row indices, pooled Test, percentile interval',
                  'primary_difference':'Weighted Macro-F1 minus Max Confidence Macro-F1',
                  'used_to_tune_methods':False}


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_verified_cache(directory,data,expected_selection,baseline):
    """Read only. No inference fallback and no calls to the old cache writer."""
    directory=Path(directory)
    cache_path=directory/'test_probabilities.npz';freeze_path=directory/'evaluation_freeze.json'
    record_path=directory/'inference_record.json'
    for path in (cache_path,freeze_path,record_path):
        if not path.is_file(): raise FileNotFoundError(f'Missing historical Test artifact: {path}')
    freeze=read_json(freeze_path);record=read_json(record_path);config=freeze['configuration']
    if freeze.get('selected_before_test_inference') is not True:
        raise ValueError('Historical selection was not frozen before Test.')
    if (config.get('selected_alpha')!=FIXED_ALPHA or config.get('selected_method')!='ensemble' or
            config.get('selection_dataset')!='validation' or config.get('probability_columns')!=LABELS):
        raise ValueError('Historical method, alpha or label order changed.')
    if any(config.get(k)!=v for k,v in expected_selection.items()):
        raise ValueError('Historical configuration does not match the frozen Validation selection.')
    ids=ordered_row_ids(data)
    if (config.get('test_sha256')!=data['sha256'] or config.get('sample_count')!=len(ids) or
            config.get('ordered_row_ids_sha256')!=hashlib.sha256('\n'.join(ids).encode()).hexdigest()):
        raise ValueError('Test text/label/source order or dataset digest differs from the freeze.')
    if (record.get('cache_sha256')!=file_sha256(cache_path) or record.get('freeze_sha256')!=file_sha256(freeze_path) or
            baseline['provenance']['inference_record']!=record or
            baseline['provenance']['evaluation_freeze_sha256']!=file_sha256(freeze_path)):
        raise ValueError('Historical Test cache/freeze integrity or baseline anchor mismatch.')
    if (baseline['dataset']['sha256']!=data['sha256'] or baseline['sample_count']!=len(ids) or
            baseline['selected_alpha']!=FIXED_ALPHA or baseline['label_order']!=LABELS):
        raise ValueError('Historical baseline dataset/mapping mismatch.')
    with np.load(cache_path,allow_pickle=False) as cache:
        if (json.loads(cache['manifest_json'].item())!=config or cache['row_ids'].tolist()!=ids or
                not np.array_equal(cache['y_true'],data['labels']) or cache['sources'].tolist()!=data['sources']):
            raise ValueError('Cache manifest/row IDs/truth/source arrays are not aligned.')
        svm=cache['svm_probs'].copy();transformer=cache['transformer_probs'].copy()
    if svm.shape!=(len(ids),3) or transformer.shape!=svm.shape:
        raise ValueError('Historical probability count/shape mismatch.')
    compute_ensemble_probabilities(svm,transformer,FIXED_ALPHA)
    return svm,transformer


def load_inputs():
    selection_path=BACKEND/'outputs/alpha_tuning/alpha_selection.json'
    frozen=load_frozen_selection(selection_path)
    if frozen['selected_alpha']!=FIXED_ALPHA or frozen['selected_method']!='ensemble':
        raise ValueError('Step 14.2 requires the existing ensemble selection with alpha 0.8.')
    if collect_fingerprints()!=frozen['step9_fingerprints']:
        raise ValueError('Original model/data/code fingerprints have changed.')
    provenance=read_json(BACKEND/'data/test_provenance.json')
    config=read_json(BACKEND/'models/transformer/config.json')
    expected=config['dataset_sha256']['test']
    fields=('sha256','svm_handoff_test_sha256','transformer_archive_test_sha256','transformer_training_config_test_sha256')
    if (provenance.get('dataset')!='test' or provenance.get('preprocessed') is not True or
            any(provenance.get(k)!=expected for k in fields)):
        raise ValueError('Test provenance does not match both handoffs and Transformer training config.')
    data=load_test_dataset(BACKEND/'data/test.csv',expected)
    baseline=read_json(HISTORICAL/'test_metrics.json')
    if not np.isclose(baseline['models']['ensemble']['macro_f1'],.8221833733,rtol=0,atol=5e-11):
        raise ValueError('Historical Ensemble F1 does not match the published 0.8221833733.')
    for name,digest in baseline['provenance']['evaluation_code_sha256'].items():
        file=BACKEND/name if name.startswith('run_') else BACKEND/'src'/name
        if file_sha256(file)!=digest: raise ValueError('Historical Test producer code changed.')
    svm,transformer=load_verified_cache(HISTORICAL,data,frozen,baseline)
    validation_dir=BACKEND/'outputs/max_confidence_vs_ensemble'
    validation_path=validation_dir/'summary.json';validation=read_json(validation_path)
    manifest=read_json(validation_dir/'artifact_manifest.json')
    if manifest.get(validation_path.relative_to(ROOT).as_posix())!=file_sha256(validation_path):
        raise ValueError('Step 14.1 Validation summary integrity mismatch.')
    for file,digest in validation['provenance']['experiment_code_sha256'].items():
        if file_sha256(ROOT/file)!=digest: raise ValueError('Step 14.1 method/producer code changed.')
    if validation['alpha']!=FIXED_ALPHA or validation['label_order']!=LABELS:
        raise ValueError('Step 14.1 methods differ from the fixed comparison.')
    return data,svm,transformer,baseline,validation,provenance


def protected_manifest():
    files=set()
    for directory in (BACKEND/'models',BACKEND/'data',BACKEND/'config',BACKEND/'src',BACKEND/'tests',
                      ROOT/'frontend/src',ROOT/'frontend/e2e'):
        files.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update(BACKEND.glob('*.py'))
    files.update(ROOT.glob('README.md'))
    files.update(p for p in (ROOT/'docs').glob('*.md') if p!=REPORT_PATH)
    for directory in (BACKEND/'outputs').iterdir():
        if directory.is_dir() and directory!=OUTPUT_ROOT:
            files.update(p for p in directory.rglob('*') if p.is_file())
    for name in ('package.json','package-lock.json','playwright.config.ts','vite.config.ts'):
        p=ROOT/'frontend'/name
        if p.is_file(): files.add(p)
    return {p.relative_to(ROOT).as_posix():file_sha256(p) for p in sorted(files)}


def write_report(path,result,rows):
    names={'max_confidence':'Max Confidence','weighted_ensemble':'Weighted Ensemble'}
    methods=tuple(names);m=result['models'];v=result['validation'];stats=result['statistical_tests']
    d=result['macro_f1_delta_weighted_minus_max'];ci=stats['paired_bootstrap']['ci_95'];mc=stats['mcnemar']
    fmt=lambda x:f'{x:.10f}'
    comparison=markdown_table(['Phương pháp','Validation Macro-F1','Test Macro-F1','Test − Validation'],
        [[names[n],fmt(v['models'][n]['macro_f1']),fmt(m[n]['macro_f1']),fmt(m[n]['macro_f1']-v['models'][n]['macro_f1'])] for n in methods])
    metric=markdown_table(['Phương pháp','Accuracy','Macro-P','Macro-R','Macro-F1','ECE'],
        [[names[n]]+[fmt(m[n][k]) for k in ('accuracy','macro_precision','macro_recall','macro_f1','ece')] for n in methods])
    classes=markdown_table(['Phương pháp','Nhãn','N','Precision','Recall','F1'],
        [[names[n],c['label'],c['support']]+[fmt(c[k]) for k in ('precision','recall','f1')] for n in methods for c in m[n]['per_class']])
    sources=markdown_table(['Nguồn','N','Phương pháp','Accuracy','Macro-P','Macro-R','Macro-F1','ECE','Sai conf ≥ 0,9'],
        [[r['source'],r['sample_count'],names[n]]+[f"{r['models'][n][k]:.8f}" for k in ('accuracy','macro_precision','macro_recall','macro_f1')]+
         [f"{r['models'][n]['confidence']['ece']:.8f}",r['models'][n]['confidence']['wrong_high_confidence_counts']['0.9']]
         for r in result['by_source'] for n in methods])
    calibration=markdown_table(['Phương pháp','Confidence trung bình','Accuracy','ECE 10 bins','Sai confidence ≥ 0,9'],
        [[names[n]]+[fmt(result['confidence_diagnostics'][n][k]) for k in ('mean_confidence','accuracy','ece')]+
         [result['confidence_diagnostics'][n]['wrong_high_confidence_counts']['0.9']] for n in methods])
    g=result['correctness_groups'];counts=result['selection_counts']
    winner=result['observed_winner_on_test']
    observed=(f"{names[winner]} có Macro-F1 cao hơn trên Test." if winner!='tie' else 'Hai phương pháp có Macro-F1 bằng nhau trên Test.')
    positive_evidence=d>0 and ci[0]>0
    practical=('Có cơ sở thực nghiệm để tiếp tục dùng Weighted Ensemble theo lựa chọn Validation đã khóa. '
               'Mức cải thiện cần đặt cạnh chi phí chạy hai mô hình và yêu cầu ứng dụng.' if d>0 else
               'Giữ nguyên lựa chọn triển khai theo Validation. Kết quả Test này cần được báo cáo trung thực và nghiên cứu tiếp trên dữ liệu mới, không chọn lại bằng Test.')
    sections=[
        '# Bước 14.2 — Kiểm chứng Max Confidence và Weighted Ensemble trên Test',
        '## 1. Mục tiêu nghiên cứu',
        'Trả lời câu hỏi: kết hợp xác suất SVM và Transformer có hiệu quả tốt hơn chọn dự đoán của mô hình có confidence cao nhất hay không? Phép kiểm chứng giữ nguyên hai quy tắc sau Bước 14.1 và dùng cùng Test 8.238 mẫu. Không train, fine-tune, hiệu chuẩn lại hoặc chọn trọng số bằng Test.',
        '## 2. Hai phương pháp được so sánh',
        '**Max Confidence:** lấy xác suất lớn nhất của từng mô hình, chọn toàn bộ phân phối của mô hình tự tin hơn. Bằng nhau chính xác thì ưu tiên SVM, không làm tròn và không dùng ngưỡng gần bằng. Tái sử dụng nguyên hàm Bước 14.1. Nếu các nhãn hòa trong một phân phối, argmax ưu tiên thứ tự negative, neutral, positive.',
        '**Weighted Ensemble:** P = 0,8 × P_SVM + 0,2 × P_Transformer cho từng nhãn, rồi chọn argmax. Alpha 0,8 và phương án ensemble đã được chọn bằng Validation từ trước. Test chỉ kiểm chứng và mô tả kết quả, không thay đổi BEST_METHOD hoặc mặc định ứng dụng.',
        '## 3. Thiết lập thực nghiệm',
        'Train 65.899 mẫu theo bàn giao, Validation 8.238 mẫu, Test 8.238 mẫu. Chỉ đọc Test đã cleaned, không làm sạch lại, không tách câu hoặc loại/shuffle mẫu. Nhãn negative=0, neutral=1, positive=2. Mỗi hàng CSV là một đơn vị đánh giá, không phải đánh giá riêng phương pháp tổng hợp đoạn nhiều câu.',
        f"Test SHA-256: `{result['provenance']['test_sha256']}`. Cache: `backend/outputs/final_evaluation/test_probabilities.npz`, SHA-256 `{result['provenance']['cache_sha256']}`.",
        'Provenance Test khớp hai CSV bàn giao và hash trong training config Transformer. Kiểm tra fingerprint model/data/code, selection Validation, evaluation_freeze, inference_record, hash toàn cache và manifest. So khớp từng row ID từ index/text/label/source, nhãn thật và nguồn theo đúng thứ tự. Cache thiếu hoặc sai thì dừng, không inference lại tự động. SVM không có training-dataset SHA trong metadata, không suy diễn bằng chứng này.',
        'Cả hai phương pháp sử dụng nguyên hai ma trận xác suất (8238,3), không sửa input. Lần này không chạy suy luận mô hình mới. Precision/Recall/F1 macro là trung bình không trọng số trên đủ ba lớp với zero_division=0. Confusion matrix có hàng là nhãn thật, cột là nhãn dự đoán. ECE dùng 10 khoảng độ rộng bằng nhau trên [0,1], khoảng cuối chứa 1. Ngưỡng confidence cao là ≥ 0,9.',
        'Trước khi xuất kết quả, kiểm tra Ensemble khớp toàn bộ bốn metric tổng, confusion matrix và per-class của Bước 10 với sai số tuyệt đối 1e-12 cho số thực. Macro-F1 cũ đầy đủ là 0,8221833733152949 (công bố 0,8221833733). Các nguồn cũng phải khớp kết quả cũ. Không đánh giá lại SVM riêng lẻ, chỉ tham chiếu các metric đã công bố khi thảo luận.',
        '## 4. Kết quả Validation và đối chiếu Test',comparison,
        f"Validation: Weighted Ensemble cao hơn {v['macro_f1_delta_weighted_minus_max']:.10f} Macro-F1. Test: Weighted − Max = {d:+.10f}, tương đương {100*d:+.6f} điểm phần trăm. {observed}",
        'Validation đã dùng để chọn alpha và phương pháp nên không phải ước lượng độc lập. Test kiểm chứng hai quy tắc cố định. So sánh độ giảm/tăng giữa hai tập giúp xem tính nhất quán, không chứng minh hiệu quả trên mọi miền dữ liệu mới. Test đã được công bố trước đây và cùng hệ dữ liệu ba nguồn, nên đây là kiểm chứng bổ sung, không phải một bộ benchmark hoàn toàn mới chưa từng xem.',
        '## 5. Kết quả Test',metric,'### Precision, Recall, F1 từng nhãn',classes,'### Confusion matrix',
    ]
    for n in methods:
        sections += [f'**{names[n]}**',markdown_table(['Thật / Dự đoán',*LABELS],[[LABELS[i],*m[n]['confusion_matrix'][i]] for i in range(3)])]
    sections += ['### Kết quả từng nguồn',sources,
        'Macro-F1 toàn Test được tính từ toàn bộ dự đoán, không lấy trung bình Macro-F1 của ba nguồn. Per-class metrics, confusion matrix và reliability bins theo nguồn được lưu đầy đủ trong summary.json.',
        '## 6. Phân tích sai khác',
        markdown_table(['Thống kê','Số mẫu'],[['Max Confidence chọn SVM',counts['svm']],['Max Confidence chọn Transformer',counts['transformer']],
            ['Hòa chính xác, đã tính trong SVM',counts['exact_ties']],['Cả hai đúng',g['both_correct']],['Chỉ Max đúng',g['max_confidence_only']],
            ['Chỉ Weighted đúng',g['weighted_ensemble_only']],['Cả hai sai',g['both_wrong']],['Khác nhãn',result['disagreements']['count']],
            ['Khác nhãn nhưng cả hai sai',result['disagreements']['both_wrong_count']]]),
        f"Bốn nhóm đúng/sai cộng lại {sum(g.values())} mẫu. Số dự đoán đúng Weighted − Max = {g['weighted_ensemble_only']-g['max_confidence_only']:+d}. Chênh lệch accuracy và Macro-F1 không phải cùng một đại lượng.",
    ]
    for title,a,b in [('A. Max Confidence đúng, Weighted Ensemble sai',True,False),
                       ('B. Weighted Ensemble đúng, Max Confidence sai',False,True),('C. Cả hai phương pháp đều sai',False,False)]:
        cases=[r for r in rows if r['max_confidence_correct']==a and r['weighted_ensemble_correct']==b]
        sections += ['### '+title,
            markdown_table(['Index','Nguồn','Văn bản','Nhãn thật','SVM conf','TF conf','Max chọn','Max nhãn','Weighted nhãn'],
                [[r['sample_index'],r['source'],r['text'][:240],r['true_label'],f"{r['svm_confidence']:.5f}",f"{r['transformer_confidence']:.5f}",
                  r['selected_model'],r['max_confidence_label'],r['weighted_ensemble_label']] for r in cases[:2]])]
        if cases:
            r=cases[0]
            sections += [f"Mẫu {r['sample_index']}: Max chọn {r['selected_model']} và dự đoán {r['max_confidence_label']}, Weighted dự đoán {r['weighted_ensemble_label']}, nhãn thật {r['true_label']}. " +
                ('Confidence cao hơn dẫn đến lựa chọn đúng ở mẫu này, còn phép trộn làm đổi nhãn sang sai.' if a else
                 'Phép trộn đưa nhãn về đúng dù mô hình tự tin hơn dự đoán sai ở mẫu này.' if b else
                 'Cả chọn theo confidence lẫn kết hợp xác suất đều chưa khắc phục được lỗi ở mẫu này. Không thể xác định chắc nguyên nhân ngôn ngữ chỉ từ một ví dụ.')]
    sections += ['Ví dụ lấy tối đa hai mẫu đầu theo index của từng nhóm, không chọn theo hướng làm đẹp kết luận. Văn bản trong bảng rút gọn tối đa 240 ký tự. Predictions.csv chứa mọi mẫu và đầy đủ xác suất, disagreements.csv chứa toàn bộ mẫu khác nhãn, kể cả khi cả hai sai. Nhóm cả hai sai có thể cùng nhãn hoặc khác nhãn. Index bắt đầu từ 0. Không dùng các ví dụ để điều chỉnh thuật toán.',
        '## 7. Đánh giá độ tin cậy và độ bất định',calibration,
        f"Max Confidence chọn Transformer {counts['transformer']}/{result['sample_count']} lần ({100*counts['transformer']/result['sample_count']:.2f}%). Trên Validation tỷ lệ này là {100*v['selection_counts']['transformer']/v['sample_count']:.2f}%. Đây là mức ưu tiên quan sát được, không tự nó chứng minh lựa chọn quá nhiều hay sai.",
        f"Confidence trung bình của SVM là {result['base_mean_confidence']['svm']:.8f}, Transformer là {result['base_mean_confidence']['transformer']:.8f}. Accuracy riêng lẻ đã công bố từ Bước 10 tương ứng {result['historical_baselines']['svm']['accuracy']:.8f} và {result['historical_baselines']['transformer']['accuracy']:.8f}. Đối chiếu này mô tả thang confidence với hiệu quả thực tế, không suy luận lại SVM.",
        'SVM dùng bộ phân loại có hiệu chuẩn sigmoid, Transformer dùng softmax. Hiệu chuẩn riêng hoặc softmax không đảm bảo confidence của hai mô hình tương đương. Max Confidence có thể ưu tiên phân phối sắc nhưng sai, trong khi Weighted giữ cả ba xác suất của cả hai mô hình. ECE đo chênh lệch confidence–accuracy theo các khoảng, không phải accuracy và không bảo đảm hiệu chuẩn cho mọi miền. Không có can thiệp hiệu chuẩn đối chứng nên không quy toàn bộ chênh lệch cho hiệu chuẩn.',
        '### Kiểm định McNemar và paired bootstrap',
        f"McNemar chính xác hai phía: chỉ Max đúng b={mc['max_only_correct']}, chỉ Weighted đúng c={mc['weighted_only_correct']}, p={mc['p_value']:.10g}. Kiểm định nhị thức b trên b+c với H0: p=0,5, mức ý nghĩa 0,05. " +
        ('Bác bỏ giả thuyết xác suất dự đoán đúng bằng nhau theo kiểm định này.' if mc['reject_equal_correctness_at_0_05'] else 'Chưa đủ bằng chứng bác bỏ xác suất dự đoán đúng bằng nhau theo kiểm định này.'),
        f"Paired bootstrap: 10.000 lần, seed=42, mỗi lần lấy N chỉ số mẫu có hoàn lại và dùng cùng chỉ số cho hai phương pháp. Tính lại Macro-F1 đủ ba lớp rồi lấy Weighted − Max. Khoảng tin cậy percentile 95%: [{ci[0]:.10f}; {ci[1]:.10f}], tương đương [{100*ci[0]:.6f}; {100*ci[1]:.6f}] điểm phần trăm.",
        ('Khoảng bootstrap nằm hoàn toàn trên 0, hỗ trợ chênh lệch Macro-F1 dương ở mức tin cậy danh nghĩa 95% theo giả định của phép bootstrap.' if positive_evidence else
         'Diễn giải khoảng bootstrap theo vị trí so với 0. Không mặc định chênh lệch Macro-F1 có ý nghĩa thống kê chỉ vì điểm ước lượng khác nhau.'),
        'McNemar kiểm tra khác biệt đúng/sai (accuracy), không phải kiểm định trực tiếp Macro-F1. Khoảng bootstrap mô tả độ bất định của Macro-F1 cho hai mô hình đã cố định. Giả định các hàng Test độc lập cùng phân phối trong hỗn hợp nguồn hiện tại. Chưa xét phụ thuộc theo người dùng/chủ đề/hội thoại, biến thiên huấn luyện hoặc thay đổi miền. Tham số thống kê được ghi trước khi đánh giá; không thay đổi để tìm kết quả có ý nghĩa. Phân tích nguồn chỉ mô tả, không có kiểm định nhiều nguồn.',
        '## 8. Thảo luận tính cần thiết của Ensemble',
        'Weighted Ensemble có thể tốt hơn vì tận dụng thông tin của cả hai mô hình, giảm ảnh hưởng của một dự đoán tự tin nhưng sai. Max Confidence loại bỏ thông tin mô hình còn lại. Tuy nhiên, Ensemble vẫn có các mẫu làm đúng thành sai và trọng số cố định không bảo đảm tối ưu từng nguồn. Cả hai phương pháp đều cần suy luận cả hai mô hình, nên Max Confidence không tự động tiết kiệm một lượt suy luận.',
        f"SVM riêng lẻ đã có Test Macro-F1 {result['historical_baselines']['svm']['macro_f1']:.10f}. Weighted Ensemble tăng {m['weighted_ensemble']['macro_f1']-result['historical_baselines']['svm']['macro_f1']:.10f} (khoảng {100*(m['weighted_ensemble']['macro_f1']-result['historical_baselines']['svm']['macro_f1']):.6f} điểm phần trăm) so với SVM. Đây chỉ là tham chiếu kết quả cũ, không đánh giá lại/chỉnh sửa SVM và không thực hiện kiểm định Weighted–SVM trong bước này.",
        'Kết hợp có thể không cần thiết nếu mô hình riêng lẻ đã đáp ứng yêu cầu, mô hình thứ hai không đóng góp đáng kể, hoặc chi phí độ trễ/bộ nhớ vượt lợi ích. Chênh lệch có ý nghĩa thống kê không đồng nghĩa lợi ích thực tế lớn. Chưa có ngưỡng cải thiện tối thiểu, ngân sách độ trễ hoặc đánh giá tác động nghiệp vụ để khẳng định mức tăng là lớn hay đủ cho mọi ứng dụng.',
        '## 9. Kết luận nghiên cứu',
        f"**Weighted Ensemble có thực sự tốt hơn Max Confidence trên Test không?** {observed} Chênh lệch quan sát được là {d:+.10f} Macro-F1 ({100*d:+.6f} điểm phần trăm). Độ bất định và kết quả kiểm định phải đọc cùng các số liệu ở mục 7.",
        f"**Nhóm có đủ cơ sở tiếp tục sử dụng Weighted Ensemble không?** {practical} Giữ nguyên lựa chọn theo Validation, không thay đổi phương pháp vì kết quả Test.",
        '## 10. Hạn chế, tái lập và hướng phát triển',
        'Kết luận giới hạn ở Test ba nguồn hiện có, mô hình/checkpoint cố định và đơn vị văn bản theo CSV. Không chứng minh ưu thế mọi nguồn mới, mọi câu hoặc cách tổng hợp đoạn văn. Giới hạn provenance SVM được giữ nguyên. Các kiểm tra hash xác nhận tính toàn vẹn của chuỗi artifact, không thay thế đánh giá chất lượng gán nhãn hoặc kiểm chứng độc lập quá trình thu thập dữ liệu.',
        'Hướng phát triển có thể là kiểm chứng ngoài miền, hiệu chuẩn trên dữ liệu phát triển độc lập và đánh giá chi phí–chất lượng theo yêu cầu ứng dụng. Đây là đề xuất, chưa triển khai Bước 15 hoặc bất kỳ thay đổi mô hình/ứng dụng nào.',
        'Chạy tại thư mục gốc: `python -m backend.run_final_method_comparison`. Runner từ chối ghi đè một lần chạy cũ. Tái lập vào thư mục mới: `python -m backend.run_final_method_comparison --run-name reproduce_seed42`. Chạy regression: `python -m unittest discover backend/tests`.',
        'Tệp kết quả trong `backend/outputs/final_method_comparison/`: summary.json, metrics.csv, per_class_metrics.csv, source_metrics.csv, predictions.csv, disagreements.csv, statistical_tests.json và verification.json. Kèm statistical_plan.json, protected_files_before.json và artifact_manifest.json. Các số thực lưu đầy đủ, bảng báo cáo làm tròn. Verification ghi trạng thái kiểm thử thực tế và kiểm tra nội dung tệp cũ không đổi.',
    ]
    path.write_text('\n\n'.join(sections)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',help='Optional new reproduction subdirectory; never overwrites results.')
    args=parser.parse_args()
    if args.run_name and not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_name):
        parser.error('run-name must contain only letters, digits, underscores or hyphens')
    output=OUTPUT_ROOT/args.run_name if args.run_name else OUTPUT_ROOT
    report=output/'FINAL_METHOD_COMPARISON.md' if args.run_name else REPORT_PATH
    if (output.exists() and any(output.iterdir())) or report.exists():
        raise FileExistsError('Output/report already exists. Use a new --run-name; no old result is overwritten.')
    before=protected_manifest()
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'statistical_plan.json',STATISTICAL_PLAN)
    data,svm,transformer,baseline,validation,provenance=load_inputs()
    result=evaluate_fixed_methods(data['labels'],svm,transformer,data['sources'],baseline['models']['ensemble'])
    for row in result['by_source']:
        old=next(x for x in baseline['sources'] if x['source']==row['source'])
        if old['n']!=row['sample_count']: raise ValueError('Historical per-source count mismatch.')
        assert_ensemble_baseline(row['models']['weighted_ensemble'],old['models']['ensemble'])
    result['validation']=validation
    result['historical_baselines']={name:baseline['models'][name] for name in ('svm','transformer')}
    # The only base-model re-use here is saved probabilities. No SVM reevaluation.
    chosen=select_max_confidence(svm,transformer)
    ensemble=compute_ensemble_probabilities(svm,transformer,FIXED_ALPHA)
    result['statistical_tests']=paired_statistics(data['labels'],chosen['probabilities'].argmax(1),ensemble.argmax(1),
        replicates=STATISTICAL_PLAN['bootstrap_replicates'],seed=STATISTICAL_PLAN['bootstrap_seed'])
    result['provenance']={'created_at_utc':datetime.now(timezone.utc).isoformat(),'cache_status':'reused',
        'test_sha256':data['sha256'],'test_provenance':provenance,'source_counts':data['source_counts'],
        'cache_sha256':file_sha256(HISTORICAL/'test_probabilities.npz'),
        'historical_baseline_sha256':file_sha256(HISTORICAL/'test_metrics.json'),
        'ordered_row_ids_sha256':hashlib.sha256('\n'.join(ordered_row_ids(data)).encode()).hexdigest(),
        'inference_calls_this_experiment':0,'retrained':False,'alpha_retuned':False,'default_method_changed':False,
        'python_version':sys.version,'versions':{n:importlib.metadata.version(n) for n in ('numpy','scikit-learn','scipy')},
        'experiment_code_sha256':{p.relative_to(ROOT).as_posix():file_sha256(p) for p in
                                  (Path(__file__),BACKEND/'src/final_method_comparison.py')}}
    rows=prediction_rows(data,svm,transformer)
    # Keep exactly the Step-14.1 selection semantics, retaining all inputs for audit.
    write_csv(output/'predictions.csv',rows)
    write_csv(output/'disagreements.csv',[r for r in rows if r['different_labels']],list(rows[0]))
    keys=('accuracy','macro_precision','macro_recall','macro_f1','mean_confidence','ece','wrong_high_confidence_count')
    write_csv(output/'metrics.csv',[{'method':name,**{k:m[k] for k in keys}} for name,m in result['models'].items()])
    write_csv(output/'per_class_metrics.csv',[{'method':name,**r} for name,m in result['models'].items() for r in m['per_class']])
    write_csv(output/'source_metrics.csv',[{'source':r['source'],'n':r['sample_count'],'method':name,
        **{k:m[k] for k in ('accuracy','macro_precision','macro_recall','macro_f1')},'ece':m['confidence']['ece'],
        'mean_confidence':m['confidence']['mean_confidence'],'wrong_high_confidence_count':m['confidence']['wrong_high_confidence_counts']['0.9']}
        for r in result['by_source'] for name,m in r['models'].items()])
    write_json(output/'summary.json',result)
    write_json(output/'statistical_tests.json',result['statistical_tests'])
    write_report(report,result,rows)
    after=protected_manifest()
    changed=sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))
    write_json(output/'protected_files_before.json',before)
    verification={'cache_integrity_verified':True,'ordered_row_ids_verified':True,'label_order_verified':True,
        'test_sample_count':len(rows),'baseline_matches_step10':True,'source_baselines_match':True,
        'protected_files_unchanged':not changed,'protected_file_count':len(before),'changed_files':changed,
        'no_retraining_or_finetuning':True,'no_new_base_model_inference':True,'alpha_fixed':.8,
        'method_rules_unchanged':True,'test_used_for_tuning':False,'regression_tests':'pending, run separately'}
    write_json(output/'verification.json',verification)
    if changed: raise ValueError(f'Protected artifacts changed: {changed}')
    artifacts=[p for p in output.iterdir() if p.is_file()]+[report]
    write_json(output/'artifact_manifest.json',{p.relative_to(ROOT).as_posix():file_sha256(p) for p in artifacts})
    print(json.dumps({'models':{n:{k:m[k] for k in keys} for n,m in result['models'].items()},
                     'delta':result['macro_f1_delta_weighted_minus_max'],'groups':result['correctness_groups'],
                     'selection':result['selection_counts'],'disagreements':result['disagreements']['count'],
                     'statistics':result['statistical_tests'],'protected_unchanged':not changed},indent=2))


if __name__=='__main__': main()

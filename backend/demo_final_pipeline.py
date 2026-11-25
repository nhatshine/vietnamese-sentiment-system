"""Real final pipeline demo; --both presents retained results twice without re-inference."""

import argparse
import json
from pathlib import Path

from .src.prediction_service import PredictionService, present_result

DEMO_SENTENCES = (
    "Sản phẩm này rất tốt.",
    "Tôi rất thất vọng vì sản phẩm quá tệ.",
    "Sản phẩm này bình thường.",
    "Camera rất đẹp nhưng pin quá yếu.",
    "Giao hàng chậm và đóng gói cẩu thả.",
)
DEMO_PARAGRAPH = "Camera rất đẹp. Nhưng pin quá yếu. Giao hàng rất nhanh!"


def _prediction_lines(title, result):
    return [title, f"Label: {result['label']}", f"Confidence: {result['confidence']:.8f}",
            "Probabilities: " + json.dumps(result['probabilities'], ensure_ascii=False)]


def render_result(response: dict, view: str = "user") -> str:
    shown = present_result(response, view)
    lines = ["=" * 48, f"VIEW: {view}", "INPUT TEXT:", shown['input_text'], ""]
    if view == "research":
        for method in ("svm", "transformer", "ensemble"):
            lines += _prediction_lines(f"{method.upper()} RESULT", shown['comparison'][method]) + [""]
    lines += _prediction_lines("KẾT QUẢ PHÂN TÍCH CẢM XÚC — FINAL RESULT", shown['final_result'])
    lines += [f"Selected method: {shown['selection_info']['selected_method']}",
              f"Selected alpha: {shown['selection_info']['selected_alpha']}"]
    if shown['sentence_count'] > 1:
        lines += [f"Document aggregation: {shown['aggregation_info']['method']} (chưa kiểm chứng tối ưu)",
                  "KẾT QUẢ TỪNG CÂU:"]
        for sentence in shown['sentences']:
            prediction = sentence['final_result']
            lines += [f"[{sentence['sentence_index']}] {sentence['sentence']} -> {prediction['label']} ({prediction['confidence']:.8f})"]
    if view == "research":
        lines += ["RESEARCH METADATA — Macro-F1 đo trên dataset, không phải confidence của input:", json.dumps(shown['research_info'],ensure_ascii=False)]
        if shown['sentence_count'] > 1:
            lines += ["TẤT CẢ DOCUMENT AGGREGATIONS:",json.dumps(shown['document_results'],ensure_ascii=False,indent=2)]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--research',action='store_true',help='Hiện ba phương án và metadata nghiên cứu')
    group.add_argument('--both',action='store_true',help='Hiện user rồi research từ cùng response, không inference lại')
    parser.add_argument('--save-json',type=Path,help='Lưu sáu canonical responses đầy đủ để đối chiếu')
    args=parser.parse_args(argv)
    service=PredictionService()
    responses=[]
    for text in (*DEMO_SENTENCES,DEMO_PARAGRAPH):
        response=service.analyze(text)
        responses.append(response)
        views=('user','research') if args.both else ('research',) if args.research else ('user',)
        for view in views:
            print(render_result(response,view))
    if args.save_json:
        args.save_json.parent.mkdir(parents=True,exist_ok=True)
        args.save_json.write_text(json.dumps(responses,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__': main()

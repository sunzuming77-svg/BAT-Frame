import argparse
import json
import os
from typing import Any, Dict, List, Optional


PRIMARY_METRICS = ['macro_f1', 'spoof_f1', 'boundary_f1', 'frame_acc']


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_metric(value: Optional[float]) -> str:
    return '-' if value is None else f'{value:.4f}'


def get_best_record(records: List[Dict[str, Any]], key: str, mode: str) -> Optional[Dict[str, Any]]:
    if not records:
        return None

    def extract_metric(record: Dict[str, Any]) -> float:
        if key == 'val_loss':
            value = safe_float(record.get('val_loss'))
        else:
            value = safe_float(record.get('metrics', {}).get(key))
        if value is None:
            return float('inf') if mode == 'min' else float('-inf')
        return value

    ranked = sorted(records, key=extract_metric, reverse=(mode == 'max'))
    best = ranked[0]
    best_value = extract_metric(best)
    if (mode == 'min' and best_value == float('inf')) or (mode == 'max' and best_value == float('-inf')):
        return None
    return best


def summarize_experiment(model_dir: str) -> Optional[Dict[str, Any]]:
    config_path = os.path.join(model_dir, 'experiment_config.json')
    records_path = os.path.join(model_dir, 'validation_records.json')
    if not (os.path.exists(config_path) and os.path.exists(records_path)):
        return None

    config = load_json(config_path)
    records = load_json(records_path)
    if not isinstance(records, list) or len(records) == 0:
        return None

    best_loss_record = get_best_record(records, 'val_loss', mode='min')
    best_macro_record = get_best_record(records, 'macro_f1', mode='max')
    best_spoof_record = get_best_record(records, 'spoof_f1', mode='max')
    best_boundary_record = get_best_record(records, 'boundary_f1', mode='max')

    summary = {
        'model_tag': config.get('model_tag', os.path.basename(model_dir)),
        'comment': config.get('comment'),
        'model_dir': model_dir,
        'num_records': len(records),
        'config': {
            'use_boundary_control': config.get('use_boundary_control'),
            'use_cross_routing': config.get('use_cross_routing'),
            'use_soft_segments': config.get('use_soft_segments'),
            'num_segments': config.get('num_segments'),
            'seg_cons_weight': config.get('seg_cons_weight'),
            'seg_entropy_weight': config.get('seg_entropy_weight'),
            'bound_sparse_weight': config.get('bound_sparse_weight'),
            'bound_sharp_weight': config.get('bound_sharp_weight'),
        },
        'best_by_val_loss': best_loss_record,
        'best_by_macro_f1': best_macro_record,
        'best_by_spoof_f1': best_spoof_record,
        'best_by_boundary_f1': best_boundary_record,
    }
    return summary


def collect_experiments(models_dir: str, comment_filter: str = '') -> List[Dict[str, Any]]:
    results = []
    if not os.path.exists(models_dir):
        return results

    for entry in sorted(os.listdir(models_dir)):
        model_dir = os.path.join(models_dir, entry)
        if not os.path.isdir(model_dir):
            continue
        summary = summarize_experiment(model_dir)
        if summary is None:
            continue
        model_tag = summary['model_tag'] or ''
        comment = summary['comment'] or ''
        if comment_filter and comment_filter not in model_tag and comment_filter not in comment:
            continue
        results.append(summary)
    return results


def to_table_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    best_loss = summary.get('best_by_val_loss') or {}
    best_metrics = best_loss.get('metrics', {})
    return {
        'model_tag': summary['model_tag'],
        'comment': summary.get('comment') or '-',
        'boundary_control': summary['config'].get('use_boundary_control'),
        'cross_routing': summary['config'].get('use_cross_routing'),
        'soft_segments': summary['config'].get('use_soft_segments'),
        'num_segments': summary['config'].get('num_segments'),
        'best_epoch_by_loss': best_loss.get('epoch'),
        'val_loss': safe_float(best_loss.get('val_loss')),
        'macro_f1': safe_float(best_metrics.get('macro_f1')),
        'spoof_f1': safe_float(best_metrics.get('spoof_f1')),
        'boundary_f1': safe_float(best_metrics.get('boundary_f1')),
        'frame_acc': safe_float(best_metrics.get('frame_acc')),
    }


def print_summary_table(summaries: List[Dict[str, Any]]) -> None:
    headers = [
        'model_tag', 'comment', 'boundary_control', 'cross_routing', 'soft_segments',
        'num_segments', 'best_epoch_by_loss', 'val_loss', 'macro_f1', 'spoof_f1', 'boundary_f1', 'frame_acc'
    ]
    rows = [to_table_row(summary) for summary in summaries]
    widths = {}
    for header in headers:
        widths[header] = len(header)
        for row in rows:
            value = row.get(header)
            if header in PRIMARY_METRICS or header == 'val_loss':
                text = format_metric(value)
            else:
                text = str(value)
            widths[header] = max(widths[header], len(text))

    header_line = ' | '.join(header.ljust(widths[header]) for header in headers)
    separator = '-+-'.join('-' * widths[header] for header in headers)
    print(header_line)
    print(separator)
    for row in rows:
        cells = []
        for header in headers:
            value = row.get(header)
            if header in PRIMARY_METRICS or header == 'val_loss':
                text = format_metric(value)
            else:
                text = str(value)
            cells.append(text.ljust(widths[header]))
        print(' | '.join(cells))


def print_best_metric_breakdown(summaries: List[Dict[str, Any]]) -> None:
    for summary in summaries:
        print('\n=== {} ==='.format(summary['model_tag']))
        print('comment: {}'.format(summary.get('comment') or '-'))
        for block_name, label in [
            ('best_by_val_loss', 'best by val_loss'),
            ('best_by_macro_f1', 'best by macro_f1'),
            ('best_by_spoof_f1', 'best by spoof_f1'),
            ('best_by_boundary_f1', 'best by boundary_f1'),
        ]:
            record = summary.get(block_name)
            if not record:
                print('  {}: -'.format(label))
                continue
            metrics = record.get('metrics', {})
            print(
                '  {} -> epoch={} val_loss={} macro_f1={} spoof_f1={} boundary_f1={} frame_acc={}'.format(
                    label,
                    record.get('epoch'),
                    format_metric(safe_float(record.get('val_loss'))),
                    format_metric(safe_float(metrics.get('macro_f1'))),
                    format_metric(safe_float(metrics.get('spoof_f1'))),
                    format_metric(safe_float(metrics.get('boundary_f1'))),
                    format_metric(safe_float(metrics.get('frame_acc'))),
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize BAT-Mamba experiment records')
    parser.add_argument('--models_dir', type=str, default='models', help='directory containing experiment folders')
    parser.add_argument('--filter', type=str, default='', help='substring filter for model_tag/comment')
    parser.add_argument('--json_out', type=str, default='', help='optional path to save aggregated summary JSON')
    args = parser.parse_args()

    summaries = collect_experiments(args.models_dir, comment_filter=args.filter)
    if not summaries:
        print('No experiment summaries found in {}'.format(args.models_dir))
        return

    print_summary_table(summaries)
    print_best_metric_breakdown(summaries)

    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump(summaries, fh, indent=2)
        print('\nSaved aggregated summary to {}'.format(args.json_out))


if __name__ == '__main__':
    main()

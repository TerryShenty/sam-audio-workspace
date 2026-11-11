"""Download the ablation data (transfer/CPU resources only).

  animal-3000 training data  public HF dataset (configs/ablation_data.json)
  General Sound benchmark    PRIVATE HF dataset at a pinned commit; needs your own HF
                             login with read access. Only metadata and the GS files are
                             fetched. Do not redistribute the benchmark.
"""
import argparse, csv, json
from pathlib import Path
P = Path(__file__).resolve().parent.parent


def main():
    cfg = json.loads((P / 'configs' / 'ablation_data.json').read_text())
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument('--destination', type=Path, required=True)
    a = a.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download
    dest = a.destination.expanduser().resolve()
    animal = cfg['animal']
    snapshot_download(repo_id=animal['repo'], repo_type='dataset', revision=animal['revision'], local_dir=dest / 'animal3000')
    bench = cfg['benchmark']
    root = dest / 'benchmark'
    meta = hf_hub_download(bench['repo'], 'metadata/prompts.csv', repo_type='dataset', revision=bench['revision'], local_dir=root)
    with open(meta, newline='', encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r.get('dataset', '').strip().lower() in set(bench['gs_dataset_names'])]
    files = sorted({r['video_path'] for r in rows} | {r['gt_audio_path'] for r in rows})
    for i, name in enumerate(files, 1):
        hf_hub_download(bench['repo'], name, repo_type='dataset', revision=bench['revision'], local_dir=root)
        if i % 100 == 0:
            print(f'{i}/{len(files)} benchmark files', flush=True)
    (dest / 'DOWNLOAD_RECEIPT.json').write_text(json.dumps(dict(animal=animal, benchmark=bench, gs_prompt_rows=len(rows),
                                                                gs_files=len(files)), indent=2) + '\n')
    print('Downloaded. Next: python scripts/ablation_prepare.py --animal', dest / 'animal3000', '--benchmark', root, '--out <DATA_ROOT>')


if __name__ == '__main__':
    main()

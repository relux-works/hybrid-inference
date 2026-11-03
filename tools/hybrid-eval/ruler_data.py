#!/usr/bin/env python3
"""GPU-host-only RULER data bootstrap and immutable SHA-256 input lock."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

from hybrid_eval import require, sha256

VENDOR = Path(__file__).with_name('vendor')/'ruler'
SOURCE = json.loads((VENDOR/'SOURCE.json').read_text())
DATA_FILES = ('PaulGrahamEssays.json', 'squad.json', 'hotpotqa.json', 'english_words.json', 'punkt_tab.zip')
PUNKT_FILES = {'tokenizers/punkt_tab/english/'+name for name in
               ('abbrev_types.txt', 'collocations.tab', 'ortho_context.tab', 'sent_starters.txt')}
PINNED = {
    'punkt_tab.zip': ('https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers/punkt_tab.zip',
                      'e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106'),
    'squad.json': ('https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json',
                   '80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8'),
    'hotpotqa.json': ('https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json',
                     'e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8'),
    'english_words.json': ('https://media.githubusercontent.com/media/NVIDIA/RULER/'+SOURCE['commit']+'/scripts/data/synthetic/json/english_words.json',
                           'affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca'),
}


def verify_source():
    for name, item in SOURCE['files'].items():
        require(sha256(VENDOR/name) == item['sha256'], 'vendored RULER source changed: '+name)


def verify_data(root):
    lock = json.loads((root/'inputs.lock.json').read_text())
    require(lock['generator_commit'] == SOURCE['commit'], 'input generator revision differs')
    require(set(lock['files']) == set(DATA_FILES), 'incomplete data lock')
    for name in DATA_FILES:
        require(sha256(root/'json'/name) == lock['files'][name], 'RULER input hash mismatch: '+name)
    require(sha256(root/'downloads.json') == lock['downloads_manifest_sha256'], 'download provenance changed')
    require(set(lock['nltk_files']) == PUNKT_FILES, 'incomplete English Punkt lock')
    for name, expected in lock['nltk_files'].items():
        require(sha256(root/'nltk_data'/name) == expected, 'NLTK input changed')
    return lock


def bootstrap(root, fetch):
    # fetch is injected only by tests; the CLI always downloads and records the actual response bytes.
    import html2text
    from bs4 import BeautifulSoup
    root.mkdir(parents=True, exist_ok=False)
    (root/'json').mkdir()
    raw = root/'downloads'
    raw.mkdir()
    records = []
    def get(url, name, expected=None):
        content, resolved = fetch(url)
        digest = hashlib.sha256(content).hexdigest()
        record = dict(url=url, resolved_url=resolved, file=name, bytes=len(content), sha256=digest)
        # Signed CDN redirect URLs can contain short-lived credentials; keep the requested public URL instead.
        record['resolved_url'] = resolved.split('?')[0]
        records.append(record)
        (root/'downloads.json').write_text(json.dumps(records, indent=2)+'\n')
        require(expected is None or digest == expected, 'download hash mismatch: '+name)
        (raw/name).write_bytes(content)
        return content
    for name, (url, digest) in PINNED.items():
        (root/'json'/name).write_bytes(get(url, name, digest))
    nltk_files = {}
    with zipfile.ZipFile(root/'json'/'punkt_tab.zip') as archive:
        for name in archive.namelist():
            if name.startswith('punkt_tab/english/') and not name.endswith('/'):
                target = root/'nltk_data'/'tokenizers'/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
                nltk_files['tokenizers/'+name] = sha256(target)
    converter = html2text.HTML2Text()
    for name, value in dict(ignore_images=True, ignore_tables=True, escape_all=True,
                            reference_links=False, mark_code=False).items():
        setattr(converter, name, value)
    repo_parts, html_parts = {}, {}
    for i, original in enumerate((VENDOR/'PaulGrahamEssays_URLs.txt').read_text().splitlines()):
        if not original.strip():
            continue
        url = original.replace('http://', 'https://')
        url = url.replace('/LLMTest_NeedleInAHaystack/raw/main/', '/LLMTest_NeedleInAHaystack/raw/021385d68d3202e37893e9d3cd29011c569abe30/')
        content = get(url, f'essay-{i:03d}.raw')
        if '.html' in original:
            soup = BeautifulSoup(content.decode('unicode_escape'), 'html.parser')
            require(soup.find('font') is not None, 'essay HTML structure changed')
            html_parts[original.rsplit('/', 1)[-1]] = converter.handle(str(soup.find('font')))
        else:
            repo_parts[original.rsplit('/', 1)[-1]] = content.decode('utf-8')
    text = ''.join(repo_parts[k] for k in sorted(repo_parts)) + ''.join(html_parts[k] for k in sorted(html_parts))
    (root/'json'/'PaulGrahamEssays.json').write_text(json.dumps(dict(text=text)))
    lock = dict(generator_commit=SOURCE['commit'], files={n: sha256(root/'json'/n) for n in DATA_FILES},
                downloads_manifest_sha256=sha256(root/'downloads.json'), nltk_files=nltk_files,
                essay_policy='NVIDIA URL list, HTTPS, sorted repo then HTML files; first bootstrap freezes raw bytes')
    (root/'inputs.lock.json').write_text(json.dumps(lock, indent=2)+'\n')
    verify_data(root)
    return lock


def download(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read(), response.url


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['bootstrap', 'verify'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--gpu-host-downloads', action='store_true', help='explicit opt-in; never use on agents1')
    args = parser.parse_args()
    verify_source()
    if args.operation == 'bootstrap':
        require(args.gpu_host_downloads, 'downloads require explicit GPU-host opt-in')
        bootstrap(args.data, download)
    else:
        verify_data(args.data)
    print(sha256(args.data/'inputs.lock.json'))

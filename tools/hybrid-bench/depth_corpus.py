"""Pinned public depth-study inputs; retrieval is called only by the GPU stage."""
import gzip
import json
from pathlib import Path
import re
import urllib.parse

import hetero_study as study

KINDS = ('code', 'chat', 'prose')
LIMIT = 64 * 1024**2


def documents(manifest):
    data = study.read(manifest)
    study.require(data.get('schema') == 'depth-corpus-1', 'unsupported corpus manifest')
    rows = data['documents']
    study.require(len(rows) == 3 and {r['kind'] for r in rows} == set(KINDS), 'corpus requires code, chat and prose observations')
    study.require(len({r['id'] for r in rows}) == 3, 'duplicate corpus ID')
    for row in rows:
        study.require(re.fullmatch(r'[A-Za-z0-9_-]+', row['id']) and row['format'] in ('text', 'oasst-trees-gzip'), 'invalid corpus ID/format')
        study.require((row['kind'] == 'chat') == (row['format'] == 'oasst-trees-gzip'), 'corpus type/format mismatch')
        study.require(row['files'], 'empty corpus files')
        for item in row['files']:
            url = urllib.parse.urlsplit(item['url'])
            revision = item['revision']
            study.require(re.fullmatch('[0-9a-f]{40}', revision) and re.fullmatch('[0-9a-f]{64}', item['sha256']) and
                          type(item['size']) is int and 0 < item['size'] <= LIMIT, 'corpus requires revision, SHA256 and bounded size')
            allowed = ('https://raw.githubusercontent.com/ggml-org/llama.cpp/' + revision + '/') if row['kind'] == 'code' else (
                'https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/' + revision + '/' if row['kind'] == 'chat' else
                'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/' + revision + '/')
            study.require(item['url'].startswith(allowed) and url.scheme == 'https' and not url.query and not url.fragment and
                          not url.username and '..' not in urllib.parse.unquote(url.path).split('/'), 'only pinned public corpus URLs are allowed')
    return rows


def conversation(tree):
    # Select the first English, non-deleted root-to-leaf path in stored order, retaining only role and text.
    node = tree.get('prompt')
    parts = []
    while node is not None:
        study.require(len(parts) < 128 and isinstance(node, dict), 'invalid conversation tree depth')
        if node.get('lang') != 'en' or node.get('deleted', False):
            break
        role = node.get('role')
        study.require(role in ('prompter', 'assistant') and isinstance(node.get('text'), str), 'invalid conversation message')
        parts.append(('User: ' if role == 'prompter' else 'Assistant: ') + node['text'])
        replies = node.get('replies', [])
        study.require(isinstance(replies, list), 'invalid conversation replies')
        node = next((r for r in replies if r.get('lang') == 'en' and not r.get('deleted', False)), None)
    return '\n'.join(parts) + '\n\n' if len(parts) >= 2 else ''


def convert(row, paths):
    parts = []
    size = 0
    for path in paths:
        if row['format'] == 'text':
            text = path.read_text(encoding='utf-8')
            # Dataset separators remain ordinary document text, not live chat-control tokens.
            text = text.replace('<|endoftext|>', '\n\n')
            parts.append(text)
            size += len(text.encode())
        else:
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                for line in stream:
                    study.require(len(line) <= 8*1024**2, 'conversation line exceeds limit')
                    text = conversation(json.loads(line))
                    size += len(text.encode())
                    study.require(size <= LIMIT, 'derived corpus exceeds limit')
                    parts.append(text)
        study.require(size <= LIMIT, 'derived corpus exceeds limit')
    text = '\n\n'.join(parts)
    study.require(text.strip() and len(text.encode()) <= LIMIT, 'empty or oversized derived corpus')
    # Corpus content is quoted as data inside the single outer user turn.
    return text.replace('<|', '< |')


def fetch(manifest, root):
    rows = documents(manifest)
    root = study.private_dir(root)
    result = []
    for row in rows:
        folder = study.private_dir(root/row['id'])
        paths = []
        for i, item in enumerate(row['files']):
            target = folder/f'source-{i}.raw'
            study.require(not target.exists(), 'corpus output exists; use a fresh root')
            study.run(['curl', '--fail', '--location', '--proto', '=https', '--proto-redir', '=https', '--retry', '3',
                       '--max-time', '600', '--max-filesize', str(item['size']), '--output', target, item['url']], folder, f'fetch-{i}', 660)
            study.require(target.stat().st_size == item['size'] and study.sha256(target) == item['sha256'], 'corpus source SHA256/size mismatch')
            paths.append(target)
        text = folder/'document.txt'
        study.atomic(text, convert(row, paths).encode())
        record = dict(row, derived_sha256=study.sha256(text), derived_size=text.stat().st_size,
                      transform='english-first-path-and-quoted-specials-v1' if row['kind'] == 'chat' else 'ordered-text-and-quoted-specials-v1',
                      path=str(text), observation_unit='one constructed document')
        study.save(folder/'identity.json', record)
        result.append(record)
    study.save(root/'manifest.json', dict(schema='depth-derived-corpus-1', documents=result))
    return result


def select(rows, depth, million_documents=2):
    # At 512k all three types run; at 1M the default keeps code and chat, with prose opt-in.
    by_kind = {r['kind']: r for r in rows}
    return [by_kind[k] for k in KINDS[:million_documents if depth > 524288 else 3]]

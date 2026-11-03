#!/usr/bin/env python3
"""Adapter for the pinned NVIDIA generators, with full rendered-prompt length control."""
import importlib.metadata
import itertools
import random
import re
import sys
import types
from pathlib import Path
from typing import Any

from hybrid_eval import digest, require
from ruler_data import VENDOR, verify_data, verify_source


def runtime_identity():
    identity = {}
    for line in Path(__file__).with_name('ruler-requirements.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            name, version = line.split('==')
            actual = importlib.metadata.version(name)
            require(actual == version, 'RULER dependency version differs: '+name)
            identity[name] = actual
    import wonderwords
    from hybrid_eval import sha256
    folder = Path(wonderwords.__file__).parent/'assets'
    identity['wordlists'] = {p.name: sha256(p) for p in sorted(folder.glob('*.txt'))}
    return identity


def configurations():
    import yaml
    return yaml.safe_load((VENDOR/'synthetic.yaml').read_text())


def postprocess(answer):
    # References contain no control characters, so this preserves upstream substring matches.
    return re.sub(r'[\x00-\x1f]', '\n', answer.strip()).strip()


def score(task, answers, observed):
    require(answers and all(isinstance(a, str) and a for a in answers), 'empty RULER reference')
    found = [a.lower() in postprocess(observed).lower() for a in answers]
    return float(any(found)) if task.startswith('qa_') else sum(found)/len(found)


def body_for(text, prefix):
    return dict(messages=[dict(role='user', content=text), dict(role='assistant', content=prefix)],
                continue_final_message=True, add_generation_prompt=False,
                chat_template_kwargs=dict(enable_thinking=False))


class Generator:
    def __init__(self, task, data, tokenize, seed, fixture=False):
        import numpy as np
        verify_source()
        runtime_identity()
        if not fixture:
            verify_data(data)
        config = configurations()[task]
        family = config['task']
        self.task, self.family, self.seed, self.np = task, family, seed, np
        constants = types.ModuleType('constants')
        exec(compile((VENDOR/'constants.py').read_bytes(), str(VENDOR/'constants.py'), 'exec'), constants.__dict__)
        self.base = constants.TASKS[family]
        arguments = ['ruler', '--save_dir', str(data), '--save_name', task, '--tokenizer_path', 'live-GGUF',
                     '--max_seq_length', '8192', '--tokens_to_generate', str(self.base['tokens_to_generate']),
                     '--num_samples', '1', '--random_seed', str(seed),
                     '--template', self.base['template']+self.base['answer_prefix']]
        for name, value in config['args'].items():
            arguments += ['--'+name, str(value)]
        tokenizer = types.ModuleType('tokenizer')
        adapter = types.SimpleNamespace(text_to_tokens=tokenize)
        setattr(tokenizer, 'select_tokenizer', lambda *args: adapter)
        manifest = types.ModuleType('manifest_utils')
        setattr(manifest, 'write_manifest', lambda *args, **kwargs: None)
        old_argv = sys.argv
        modules = {key: sys.modules.get(key) for key in ('constants', 'tokenizer', 'manifest_utils')}
        namespace: dict[str, Any] = dict(__name__='ruler_generator', __file__=str(data/(family+'.py')))
        try:
            sys.argv = arguments
            sys.modules.update(constants=constants, tokenizer=tokenizer, manifest_utils=manifest)
            exec(compile((VENDOR/(family+'.py')).read_bytes(), str(VENDOR/(family+'.py')), 'exec'), namespace)
        finally:
            sys.argv = old_argv
            for key, value in modules.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value
        self.module = namespace
        self.icl_example = None
        if family == 'variable_tracking':
            self.icl_example = namespace['sys_vartrack_w_noise_random'](
                num_samples=1, max_seq_length=500, incremental=5,
                num_chains=namespace['args'].num_chains, num_hops=namespace['args'].num_hops)[0]

    def candidate(self, index, size, depth):
        seed = int(digest([self.seed, self.task, index])[:8], 16)
        random.seed(seed)
        self.np.random.seed(seed)
        module = self.module
        # Match upstream's short-context CWE policy, without invoking its unbounded retry loops.
        module['args'].max_seq_length = depth
        family = self.family
        function = module['generate_input_output']
        if family == 'variable_tracking':
            text, answers = function(size, module['args'].num_chains, module['args'].num_hops)
            example = self.icl_example
            assert example is not None
            worked = example['input'] + ' ' + ' '.join(example['outputs']) + '\n'
            cutoff = text.index(self.base['template'][:20])
            return text[:cutoff] + module['randomize_icl'](worked) + '\n' + text[cutoff:], answers
        if family == 'freq_words_extraction':
            return function(max_len=depth*100, num_words=size, vocab_size=max(4, (depth-self.base['tokens_to_generate'])//50),
                            alpha=module['args'].alpha)[:2]
        if family == 'qa':
            # A permutation chooses distinct QA rows; per-example RNG still controls distractor positions.
            indices = list(range(len(module['QAS'])))
            random.Random(self.seed).shuffle(indices)
            require(index < len(indices), 'not enough distinct QA examples')
            return function(indices[index], size)
        return function(size)

    def generate(self, index, depth, render_tokens):
        require(512 <= depth <= 1048576 and index >= 0, 'invalid RULER depth or index')
        minimum = 10 if self.family == 'common_words_extraction' else 5 if self.family == 'variable_tracking' else 4 if self.family == 'niah' else 10
        if self.family == 'qa':
            indices = list(range(len(self.module['QAS'])))
            random.Random(self.seed).shuffle(indices)
            require(index < len(indices), 'not enough distinct QA examples')
            q = self.module['QAS'][indices[index]]
            minimum = len(q['context'])
        def candidate(size):
            text, answers = self.candidate(index, size, depth)
            marker = self.base['answer_prefix'][:10]
            at = text.rfind(marker)
            require(at >= 0, 'NVIDIA answer prefix missing')
            prompt, prefix = text[:at], text[at:]
            return prompt, prefix, answers, render_tokens(body_for(prompt, prefix))
        first = candidate(minimum)
        require(len(first[3]) <= depth, 'RULER task minimum exceeds context depth')
        # Upstream estimates tokens per size unit and allows 3x slack, including for document units.
        lo, hi = minimum, max(minimum+1, int(3*depth*minimum/max(1, len(first[3]))))
        capacity = None
        if self.family == 'common_words_extraction':
            capacity = max(len(self.module['words']), len(self.module['randle_words']))
            hi = min(hi, capacity)
        # A bounded deterministic search changes only the upstream haystack-size argument.
        for _ in range(24):
            if len(candidate(hi)[3]) > depth:
                break
            if capacity is not None and hi == capacity:
                break
            hi *= 2
            if capacity is not None:
                hi = min(hi, capacity)
        else:
            require(False, 'RULER generator length did not grow')
        best = first
        while lo <= hi:
            mid = (lo+hi)//2
            value = candidate(mid)
            if len(value[3]) <= depth:
                best, lo = value, mid+1
            else:
                hi = mid-1
        prompt, prefix, answers, tokens = best
        # Whitespace pads framing only; it never truncates a task, needle, document or query.
        padding = ''
        low, high = 0, 2*(depth-len(tokens))+16
        while len(tokens) < depth and low <= high:
            mid = (low+high)//2
            candidate_padding = ' \t\n'*mid
            trial = best[0].replace('\n', '\n'+candidate_padding, 1)
            padded = render_tokens(body_for(trial, prefix))
            if len(padded) <= depth:
                prompt, tokens, padding, low = trial, padded, candidate_padding, mid+1
            else:
                high = mid-1
        units = [''.join(chars) for size in range(1, 5) for chars in itertools.product(' \t\n', repeat=size)]
        for _ in range(depth-len(tokens)+16):
            if len(tokens) == depth:
                break
            for unit in units:
                trial = best[0].replace('\n', '\n'+padding+unit, 1)
                padded = render_tokens(body_for(trial, prefix))
                if len(tokens) < len(padded) <= depth:
                    prompt, tokens, padding = trial, padded, padding+unit
                    break
            else:
                require(False, f'tokenizer cannot reach exact depth with whitespace padding: {self.task} {len(tokens)}/{depth}')
        require(len(tokens) == depth, 'RULER rendered depth differs')
        positions = [m.start() for m in re.finditer(r'One of the special magic .*? is: .*?\.|VAR [A-Z]+ =[^\n.]*|Document \d+:', prompt)]
        return dict(depth=depth, task=self.task, index=index, seed=self.seed, input=prompt, answer_prefix=prefix,
                    answers=answers, max_tokens=self.base['tokens_to_generate'], prompt_sha256=digest(tokens),
                    input_sha256=digest([prompt, prefix]), prompt_tokens=len(tokens),
                    padding_characters=len(prompt)-len(best[0]),
                    record_character_offsets=positions,
                    position_policy='NVIDIA generator random positions; frozen prompt shared by all arms')

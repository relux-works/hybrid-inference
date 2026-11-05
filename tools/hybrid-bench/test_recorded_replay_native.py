#!/usr/bin/env python3
"""Opt-in CPU controls for native one-token-at-a-time recorded output replay."""
import argparse
import json
from pathlib import Path
import sys
from bench_common import require
from session_measurement import native_stream
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hybrid-proxy'))
from native_fixture import native_server
from proxy import HTTPError


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('build-dir','model','artifacts'):parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args();root=args.artifacts.resolve();root.mkdir(mode=0o700)
    with native_server(args.build_dir.resolve()/'bin/llama-server',args.model.resolve(),root,checkpoints=0) as (decoder,_):
        prompt=decoder.request('POST','/tokenize',dict(content='The clear sky above the meadow.',add_special=True))['tokens']
        baseline=dict(prompt=prompt,n_predict=4,temperature=0,return_tokens=True,cache_prompt=True,ignore_eos=True)
        natural=decoder.request('POST','/completion',baseline)['tokens']
        forced=[100,101,102,103]
        require(natural!=forced,'non-greedy replay control was not applied')
        request=dict(baseline,replay_mode='recorded-token-replay-v1',replay_tokens=forced)
        generated,metrics,_,_=native_stream(decoder,request)
        require(generated==forced,'native replay failed to emit recorded IDs')
        state=decoder.request('GET','/slots/0/info')
        require(state['tokens']==(prompt+forced)[:-1],'native replay evaluated a different output journal')
        require(metrics['predicted_n']==4,'replay did not use native decode counter')
        eos=decoder.request('POST','/tokenize',dict(content='<|im_end|>',add_special=False,parse_special=True))['tokens']
        require(len(eos)==1,'special-token control must resolve one token')
        with_eos=[100,eos[0],101,103]
        replay_eos=dict(request,replay_tokens=with_eos)
        emitted,_,_,_=native_stream(decoder,replay_eos)
        require(emitted==with_eos,'recorded EOS incorrectly ended the replay early')
        state=decoder.request('GET','/slots/0/info')
        rejected=[]
        # Invalid replay contracts must refuse before changing the previously evaluated state.
        cases=[dict(replay_mode='other'),dict(replay_tokens=[True]*4),dict(replay_tokens=[-1]*4),
               dict(replay_tokens=[999999999]*4),dict(n_predict=3),dict(stop=['x']),dict(ignore_eos=False),dict(grammar='root ::= \"x\"'),dict(n_probs=1),dict(reasoning_budget_tokens=0)]
        for mutant in cases:
            try:decoder.request('POST','/completion',dict(request,**mutant))
            except HTTPError as error:require(error.status==400,'unrelated native replay refusal')
            else:raise AssertionError('invalid replay contract accepted')
            require(decoder.request('GET','/slots/0/info')['tokens']==state['tokens'],'refused replay changed native state')
            rejected.append(list(mutant)[0])
        again=decoder.request('POST','/completion',baseline)['tokens']
        require(again==natural,'default greedy path changed after recorded replay')
        result=dict(schema='recorded-replay-cpu-v1',native_decode_positions=metrics['predicted_n'],
            forced_non_greedy=True,evaluated_journal_exact=True,recorded_eos_does_not_stop=True,rejected=rejected,default_greedy_unchanged=True)
        (root/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

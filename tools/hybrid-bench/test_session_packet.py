"""Device fixture/projection controls for the r49 operator packet."""
import importlib.util
import json
from pathlib import Path
import unittest
from bench_common import Refusal
from growing_fixture import validate_fixture
from session_runtime import select_workload
from test_session_measurement import export_fixture,MODEL

PACKET=Path(__file__).parent/'packets/r49'

def module(name):
    spec=importlib.util.spec_from_file_location('r49_'+name,PACKET/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


class PacketTests(unittest.TestCase):
    # Each delivered fixture has a measured PRI31 source and distinct device identity; placeholders must still refuse.
    def test_device_fixtures_are_measured_and_bound(self):
        hashes=set()
        for filename,device in [('m3-max-qwen-pri31.json','m3-max-128gb'),('m5-max-qwen-pri31.json','m5-max-64gb')]:
            fixture=json.loads((PACKET/'fixtures'/filename).read_text())
            self.assertEqual(fixture['target_device'],device)
            self.assertTrue(validate_fixture(fixture,fixture['model_sha256'],'qwen35'))
            hashes.add(fixture['provenance']['archive_sha256'])
            with self.assertRaisesRegex(Refusal,'placeholder/background'):
                validate_fixture(dict(fixture,status='placeholder'),fixture['model_sha256'],'qwen35')
        self.assertEqual(len(hashes),2)

    # The estimator preserves an empty-arm negative and charges one fixed state for every forced-control import.
    def test_estimates_empty_and_nonempty_controls(self):
        estimate=module('estimate').estimate
        workload=select_workload(export_fixture(),MODEL,10)
        fixture=dict(depths=[dict(depth=0,decode_tok_s=10)])
        config=dict(local_rates=[[0,1e9]],server_rate=1000,link_rate=20e6,kv_bytes=1,state_bytes=1000)
        local=estimate(workload,fixture,config)
        self.assertEqual(local['arms']['hybrid']['turn_offloads'],0)
        self.assertEqual(local['arms']['hybrid']['predicted_envelope_bytes'],0)
        remote=estimate(workload,fixture,dict(config,always=True))
        self.assertEqual(remote['arms']['hybrid']['turn_offloads'],40)
        self.assertGreater(remote['arms']['hybrid']['predicted_envelope_bytes'],40*1000)
        self.assertEqual(remote['arms']['local']['predicted_envelope_bytes'],0)


if __name__=='__main__':unittest.main()

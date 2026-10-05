"""Recorded callback regression and strict fault gates. No vendor imports/robot."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import neutral_v050_nrt_runner as n
import smartjoint_csv_nrt_runner as s
from test_neutral_v050_nrt import SDK

FIXTURE = Path(__file__).parent/'fixtures/smartjoint_event_order'


class RecordedEventOrderTests(unittest.TestCase):
    def setUp(self):
        self.meta = json.loads((FIXTURE/'provenance.json').read_text())
        for name, digest in self.meta['sha256'].items():
            self.assertEqual(hashlib.sha256((FIXTURE/name).read_bytes()).hexdigest(), digest)
        self.events = [json.loads(line) for line in (FIXTURE/'motion_events.jsonl').read_text().splitlines()]
        self.batches = json.loads((FIXTURE/'append_batches.json').read_text())
        self.c = s.BatchCompletion(self.batches, self.meta['tag'], self.meta['start_ns'])

    def event(self, index, remark='', error=None, reach=True, cmd_id='absj#0'):
        return {'host_ns': self.meta['start_ns']+1,
                'payload': {'customInfo': f"{self.meta['tag']}:k{index:03d}",
                            'wayPointIndex': index, 'cmdID': cmd_id, 'remark': remark,
                            'error': {'ec': 0, 'message': 'success'} if error is None else error,
                            'reachTarget': reach}}

    def test_exact_recorded_warning_one_then_completion_zero(self):
        original = copy.deepcopy(self.events)
        self.assertEqual(len(self.events), 2)
        self.assertEqual([e['payload']['wayPointIndex'] for e in self.events], [1, 0])
        self.assertEqual(self.events[0]['payload']['remark'], 'adjacent points')
        self.assertTrue(self.events[0]['payload']['reachTarget'])
        self.c.observe(self.events[0])
        self.assertEqual(self.c.last_index, -1)
        self.assertEqual(self.c.seen, set())
        self.assertIsNone(self.c.terminal_ns)
        self.c.observe(self.events[1])
        self.assertEqual(self.c.last_index, 0)
        self.assertEqual(self.c.seen, {(0, True)})
        self.assertEqual(self.c.policy_failures, [])
        self.assertEqual(len(self.c.diagnostic_warnings), 1)
        self.assertFalse(self.c.hard_execution_fault)
        self.assertEqual(self.events, original)

    def test_normal_monotonic_completion_events(self):
        for i in range(6):
            self.c.observe(self.event(i))
        self.assertEqual(self.c.last_index, 5)
        self.assertEqual(self.c.seen, {(i, True) for i in range(6)})
        self.assertFalse(self.c.hard_execution_fault)

    def test_completion_five_then_four_remains_fatal(self):
        self.c.observe(self.event(5))
        with self.assertRaisesRegex(n.ValidationError, 'execution-progress index regressed'):
            self.c.observe(self.event(4))
        self.assertTrue(self.c.hard_execution_fault)
        self.assertEqual(self.c.last_index, 5)

    def test_invalid_indices_remain_fatal_with_warning_or_error(self):
        for index in (-1, 100, True, '1', 1.5):
            for error in ({'ec': 0, 'message': 'success'}, {'ec': 7, 'message': 'failure'}):
                c = s.BatchCompletion(self.batches, self.meta['tag'], self.meta['start_ns'])
                bad = self.event(1, remark='adjacent points', error=error)
                bad['payload']['wayPointIndex'] = index
                with self.subTest(index=index, error=error), self.assertRaises(n.ValidationError):
                    c.observe(bad)
                self.assertTrue(c.hard_execution_fault)

    def test_hard_controller_error_with_warning_remains_fatal(self):
        with self.assertRaises(n.ValidationError):
            self.c.observe(self.event(1, remark='adjacent points', error={'ec': 7, 'message': 'failure'}))
        self.assertTrue(self.c.hard_execution_fault)
        self.assertEqual(self.c.last_index, -1)

    def test_invalid_id_tag_and_malformed_warning_remain_fatal(self):
        for key, value in (('cmdID', 'unknown'), ('customInfo', 'wrong'), ('reachTarget', 1),
                           ('remark', None), ('error', None)):
            c = s.BatchCompletion(self.batches, self.meta['tag'], self.meta['start_ns'])
            bad = self.event(1, remark='adjacent points')
            bad['payload'][key] = value
            with self.subTest(key=key), self.assertRaises(n.ValidationError): c.observe(bad)
            self.assertTrue(c.hard_execution_fault)

    def test_warnings_neither_advance_nor_regress_progress_or_duplicate_state(self):
        self.c.observe(self.event(5))
        for i in (9, 1, 5, 1):
            self.c.observe(self.event(i, remark='adjacent points'))
        self.assertEqual(self.c.last_index, 5)
        self.assertEqual(self.c.seen, {(5, True)})
        self.assertEqual(self.c.policy_failures, [])
        self.assertEqual(len(self.c.diagnostic_warnings), 4)
        self.assertFalse(self.c.hard_execution_fault)
        with self.assertRaises(n.ValidationError): self.c.observe(self.event(5))

    def test_terminal_warning_cannot_complete_but_later_completion_can(self):
        p = {'customInfo': self.meta['tag']+':k507', 'wayPointIndex': 7, 'cmdID': 'absj#5',
             'remark': 'adjacent points', 'error': {'ec': 0, 'message': 'success'}, 'reachTarget': True}
        self.c.observe({'host_ns': self.meta['start_ns']+1, 'payload': p})
        self.assertIsNone(self.c.terminal_ns)
        with self.assertRaises(n.ValidationError): self.c.completed_status()
        self.c.observe({'host_ns': self.meta['start_ns']+2, 'payload': {**p, 'remark': ''}})
        self.assertEqual(self.c.completed_status(), 'COMPLETED_ACCEPTED')

    def test_false_reach_target_progress_remains_strict(self):
        self.c.observe(self.event(5, reach=False))
        with self.assertRaises(n.ValidationError): self.c.observe(self.event(4))

    def test_exact_sequence_through_existing_fifo_drain_is_logged_and_surfaced(self):
        data = s.load_csv()
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()) as output:
            store = s.CsvStore(Path(temp)/'session', data)
            store.batches = self.batches
            session = s.CsvSession(SDK, store, data, s.SOURCE, 1)
            session.completion = self.c
            for event in self.events:
                session.inbox.items.put_nowait(copy.deepcopy(event))
            session.drain_events()
            self.assertIsNone(session.robot)
            store.close()
            logged = [json.loads(line) for line in (store.folder/'motion_events.jsonl').read_text().splitlines()]
            lifecycle = [json.loads(line) for line in (store.folder/'lifecycle.jsonl').read_text().splitlines()]
        self.assertEqual([e['payload'] for e in logged], [e['payload'] for e in self.events])
        self.assertEqual(self.c.last_index, 0)
        self.assertEqual(lifecycle[0]['code'], 'ADJACENT_POINTS_DIAGNOSTIC')
        self.assertIn('WARNING:', output.getvalue())
        self.assertIn("'adjacent points'", output.getvalue())


if __name__ == '__main__': unittest.main()

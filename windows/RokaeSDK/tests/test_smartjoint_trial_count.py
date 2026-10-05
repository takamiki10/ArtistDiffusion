"""Trial count and reservation gates; SDK import and robot construction prohibited."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import neutral_v050_nrt_runner as n
import smartjoint_csv_nrt_runner as s


class ReservationReached(Exception):
    pass


class TrialCountTests(unittest.TestCase):
    def invoke(self, root, trial=6, confirm=True):
        argv=['--run-trial','--trial',str(trial)]
        if confirm: argv.append('--confirm-native-radians-seconds')
        with patch.object(s,'SESSIONS',root), \
             patch.object(n,'load_exact_sdk_data_only',side_effect=AssertionError('SDK forbidden')), \
             patch.object(s,'CsvSession',side_effect=AssertionError('robot session forbidden')), \
             patch.object(s,'CsvStore',side_effect=ReservationReached):
            s.main(argv)

    def prior_trials(self, root):
        for trial in range(1,6):
            folder=root/f'prior_{trial}';folder.mkdir()
            (folder/'final_status.json').write_text(json.dumps({
                'status':'COMPLETED_ACCEPTED','source_csv_sha256':s.SOURCE_SHA256}))
            (root/f'trial_{trial:02d}.json').write_text(json.dumps({'folder':str(folder)}))

    def test_six_is_allowed_after_five_accepted_trials(self):
        self.assertEqual(s.MAX_TRIALS,6)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.prior_trials(root)
            with self.assertRaises(ReservationReached): self.invoke(root)
            marker=json.loads((root/'trial_06.json').read_text())
            self.assertEqual(marker['trial'],6)
            self.assertEqual(marker['source_csv_sha256'],s.SOURCE_SHA256)
            self.assertTrue(Path(marker['folder']).name.startswith('trial_06_'))

    def test_missing_failed_or_different_source_prior_still_blocks_six(self):
        for prior in range(1,6):
            for condition in ('missing','failed','different_source'):
                with self.subTest(prior=prior,condition=condition), tempfile.TemporaryDirectory() as temp:
                    root=Path(temp);self.prior_trials(root)
                    if condition=='missing': (root/f'trial_{prior:02d}.json').unlink()
                    else:
                        (root/f'prior_{prior}'/'final_status.json').write_text(json.dumps({
                            'status':'FAILED_EXECUTION' if condition=='failed' else 'COMPLETED_ACCEPTED',
                            'source_csv_sha256':'wrong' if condition=='different_source' else s.SOURCE_SHA256}))
                    with self.assertRaises(n.ValidationError): self.invoke(root)
                    self.assertFalse((root/'trial_06.json').exists())

    def test_existing_six_reservation_still_blocks_duplicate_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.prior_trials(root)
            marker=root/'trial_06.json';marker.write_bytes(b'preserved')
            with self.assertRaises(FileExistsError): self.invoke(root)
            self.assertEqual(marker.read_bytes(),b'preserved')

    def test_seven_is_not_authorized_by_six_trial_extension(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised: self.invoke(Path(temp),7)
            self.assertEqual(raised.exception.code,2)

    def test_confirmation_error_message_includes_six(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(n.ValidationError,'1..6'):
                self.invoke(Path(temp),confirm=False)


if __name__=='__main__': unittest.main()

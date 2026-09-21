import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from speech_emotion import menu

SESSIONS = Path('no-such-sessions-dir')


def drive(answers):
    """Run the menu against scripted answers, returning the argv it would have run."""
    commands = []
    with patch('builtins.input', side_effect=answers), \
         patch('speech_emotion.menu.run_command', side_effect=commands.append), \
         contextlib.redirect_stdout(io.StringIO()):
        menu.run(SESSIONS)
    return commands


def choose_model(answers, model=None, backend=None):
    with patch('builtins.input', side_effect=answers), contextlib.redirect_stdout(io.StringIO()):
        return menu.choose_model(model, backend)


class MenuTests(unittest.TestCase):
    def test_quit_runs_nothing(self):
        self.assertEqual(drive(['q']), [])

    def test_unknown_choice_reprompts(self):
        self.assertEqual(drive(['9', '', 'q']), [])

    def test_end_of_input_exits(self):
        with patch('builtins.input', side_effect=EOFError), contextlib.redirect_stdout(io.StringIO()):
            menu.run(SESSIONS)

    def test_record_passes_name_and_duration(self):
        self.assertEqual(drive(['1', '5', 'morning', 'q']),
                         [['record', '--seconds', '5', '--sessions-dir', str(SESSIONS), '--name', 'morning']])

    def test_record_defaults_to_sixty_seconds_and_generated_name(self):
        self.assertEqual(drive(['1', '', '', 'q']),
                         [['record', '--seconds', '60', '--sessions-dir', str(SESSIONS)]])

    def test_review_and_analyze_need_existing_files(self):
        # Nothing exists to read or reuse, so no command should run.
        self.assertEqual(drive(['2', 'q']), [])
        self.assertEqual(drive(['3', '1', 'q']), [])
        self.assertEqual(drive(['3', '2', 'no-such-file.wav', 'q']), [])

    def test_selected_model_reaches_the_command(self):
        with tempfile.TemporaryDirectory() as bundle:
            commands = drive(['4', '2', bundle, '1', '10', 'named', 'q'])
            self.assertEqual(commands, [['record', '--seconds', '10', '--sessions-dir', str(SESSIONS),
                                         '--name', 'named', '--model', bundle]])

    def test_bundled_model_clears_a_previous_choice(self):
        self.assertEqual(choose_model(['1'], model='old-bundle'), (None, None))

    def test_missing_bundle_keeps_the_current_model(self):
        self.assertEqual(choose_model(['2', 'no-such-bundle'], model='old-bundle'), ('old-bundle', None))

    def test_backend_requires_module_and_factory(self):
        self.assertEqual(choose_model(['3', 'missing_the_colon']), (None, None))

    def test_backend_selection_records_factory(self):
        self.assertEqual(choose_model(['3', 'my_module:create', '']), (None, 'my_module:create'))


if __name__ == '__main__':
    unittest.main()

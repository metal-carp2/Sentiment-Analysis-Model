"""Interactive menu that drives the same commands the command line exposes."""
from pathlib import Path

from .cli import main as run_command


def ask(text):
    try:
        # PowerShell prefixes piped stdin with a byte order mark; strip it with the whitespace.
        return input(text).strip('﻿ \t\r\n')
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def run_command_quietly(argv):
    # main() reports its own errors and exits; keep the menu alive either way.
    try:
        run_command(argv)
    except SystemExit:
        pass


def sessions_with(sessions_dir, filename):
    if not sessions_dir.is_dir():
        return []
    return sorted((path for path in sessions_dir.iterdir() if (path/filename).is_file()), key=lambda path: path.name)


def choose(options, title, label=str):
    print(f'\n{title}')
    for index, option in enumerate(options, 1):
        print(f'  {index}) {label(option)}')
    answer = ask('Number, or blank to go back: ')
    if not answer:
        return None
    if not answer.isdigit() or not 1 <= int(answer) <= len(options):
        print('That is not one of the listed numbers.')
        return None
    return options[int(answer)-1]


def model_arguments(model, backend):
    arguments = []
    if model:
        arguments += ['--model', model]
    if backend:
        arguments += ['--backend', backend]
    return arguments


def describe_model(model, backend):
    if backend:
        return f'{backend} (custom backend)' + (f' with {model}' if model else '')
    return f'{model} (model bundle)' if model else 'bundled model'


def choose_model(model, backend):
    print('\n  1) Bundled model')
    print('  2) A TensorFlow SavedModel bundle directory')
    print('  3) A custom backend, as module:factory')
    answer = ask('Number, or blank to go back: ')
    if answer == '1':
        return None, None
    if answer == '2':
        path = ask('Path to the model bundle directory: ')
        if not path:
            return model, backend
        if not Path(path).is_dir():
            print('That directory does not exist.')
            return model, backend
        return path, None
    if answer == '3':
        print('A backend is imported and called, so it runs Python code. Only use one you trust.')
        target = ask('Backend as module:factory: ')
        if not target:
            return model, backend
        if ':' not in target:
            print('A backend must be written as module:factory.')
            return model, backend
        path = ask('Path to pass the factory, or blank for none: ')
        return path or None, target
    return model, backend


def analyze_source(sessions_dir):
    recordings = sessions_with(sessions_dir, 'audio.wav')
    print('\n  1) A recording from a past session')
    print('  2) A WAV file somewhere else')
    answer = ask('Number, or blank to go back: ')
    if answer == '1':
        if not recordings:
            print('No past recordings yet. Record a session first.')
            return None
        chosen = choose(recordings, 'Past recordings:', lambda path: path.name)
        return chosen/'audio.wav' if chosen else None
    if answer == '2':
        path = ask('Path to a WAV file: ')
        if not path:
            return None
        if not Path(path).is_file():
            print('That file does not exist.')
            return None
        return Path(path)
    return None


def run(sessions_dir=Path('emotion-sessions')):
    model = backend = None
    while True:
        recordings = sessions_with(sessions_dir, 'audio.wav')
        reports = sessions_with(sessions_dir, 'report.json')
        print('\n' + '='*60)
        print('Speech Emotion Sessions')
        print(f'Recordings folder: {sessions_dir.resolve()}')
        print(f'Saved sessions:    {len(recordings)}')
        print(f'Model:             {describe_model(model, backend)}')
        print('='*60)
        print('  1) Record a new session')
        print('  2) Read a past session report')
        print('  3) Analyze a recording')
        print('  4) Choose a model')
        print('  q) Quit')
        answer = ask('Select: ')
        if answer is None or answer.lower() in {'q', 'quit', 'exit'}:
            print('Goodbye.')
            return
        if answer == '1':
            seconds = ask('Seconds to record, 1-60, blank for 60: ') or '60'
            name = ask('Session name, blank to generate one: ')
            argv = ['record', '--seconds', seconds, '--sessions-dir', str(sessions_dir)]
            if name:
                argv += ['--name', name]
            run_command_quietly(argv + model_arguments(model, backend))
        elif answer == '2':
            if not reports:
                print('No finished sessions yet.')
                continue
            chosen = choose(reports, 'Past sessions:', lambda path: path.name)
            if chosen:
                run_command_quietly(['report', chosen.name, '--sessions-dir', str(sessions_dir)])
        elif answer == '3':
            wav = analyze_source(sessions_dir)
            if wav:
                name = ask('Name for the new session, blank to generate one: ')
                argv = ['analyze', str(wav), '--sessions-dir', str(sessions_dir)]
                if name:
                    argv += ['--name', name]
                run_command_quietly(argv + model_arguments(model, backend))
        elif answer == '4':
            model, backend = choose_model(model, backend)
        else:
            print('Choose 1, 2, 3, 4 or q.')

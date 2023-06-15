from optuna.storages import JournalFileStorage, JournalStorage
import argparse


def main():
    # Read in command line arguments for the source and destination study names
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_study', '-s', type=str, required=True)
    parser.add_argument('--source_study_name', type=str, required=True, default='tictactoe_lightning')
    parser.add_argument('--dest_study', '-d', type=str, required=True)
    parser.add_argument('--dest_study_name', type=str, required=True, default='tictactoe_lightning')
    args, _ = parser.parse_known_args()

    # Load the source study
    import optuna
    source_storage = JournalStorage(JournalFileStorage(args.source_study))
    source_study = optuna.load_study(study_name=args.source_study_name, storage=source_storage, load_if_exists=True)

    # Create the destination study
    dest_storage = JournalStorage(JournalFileStorage(args.dest_study))
    dest_study = optuna.create_study(study_name=args.dest_study_name, storage=dest_storage, load_if_exists=True)

    # Copy the trials from the source to the destination if they are completed
    n_copied = 0
    n_skipped = 0
    for trial in source_study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            dest_study.add_trial(trial)
            n_copied += 1
        else:
            n_skipped += 1

    print(f'Copied {n_copied} trials from {args.source_study} to {args.dest_study} and skipped {n_skipped} trials.')


if __name__ == '__main__':
    main()

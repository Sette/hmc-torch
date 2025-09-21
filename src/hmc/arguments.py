import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train a Hierarchical Multi-label Classification model."
    )
    # Dataset parameters
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["seq_GO", "derisi_GO", "gasch1_GO", "cellcycle_FUN"],
        help="List with dataset names.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to data and metadata files.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save models.",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        required=False,
        help="n_trials for hpo.",
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        required=False,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["csv", "torch", "arff"],
        default="arff",
        metavar="DATASET_TYPE",
        required=False,
        help="Type of dataset to load.",
    )

    parser.add_argument(
        "--non_lin",
        type=str,
        default="relu",
        choices=["relu", "tanh", "sigmoid"],
        metavar="NON_LIN",
        required=False,
        help="Non-linearity function.",
    )

    # Hardware and execution parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        metavar="DEVICE",
        required=False,
        help='Device to use (e.g., "cpu" or "cuda").',
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        metavar="EPOCHS",
        required=False,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    # Method parameters
    parser.add_argument(
        "--method",
        type=str,
        default="global",
        choices=[
            "global",
            "local",
            "globalLM",
            "global_baseline",
            "local_constrained",
            "local_mask",
            "local_test",
        ],
        metavar="METHOD",
        required=False,
        help="Method type to use.",
    )

    # Hyperparameter Optimization (HPO) parameters
    parser.add_argument(
        "--hpo",
        type=str,
        default="false",
        choices=["true", "false"],
        metavar="HPO",
        required=False,
        help="Enable or disable Hyperparameter Optimization (HPO). \
            Use 'true' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--active_levels",
        type=int,
        nargs="+",
        default=None,
        required=False,
        metavar="ACTIVE_LEVELS",
    )

    # HPO result parameters (used when HPO is disabled)
    parser.add_argument(
        "--lr_values",
        type=float,
        nargs="+",
        required=False,
        help="List of values for the learning \
            rate (used when HPO is disabled).",
    )
    parser.add_argument(
        "--dropout_values",
        type=float,
        nargs="+",
        required=False,
        metavar="DROPOUT",
        help="List of values for dropout \
            (used when HPO is disabled).",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        required=False,
        metavar="HIDDEN_DIMS",
        help="List of values for the number"
        " of hidden neurons (used when HPO is disabled).",
    )
    parser.add_argument(
        "--num_layers_values",
        type=int,
        nargs="+",
        required=False,
        metavar="NUM_LAYERS",
        help="List of values for the number of \
            layers (used when HPO is disabled).",
    )
    parser.add_argument(
        "--weight_decay_values",
        type=float,
        nargs="+",
        required=False,
        metavar="WEIGHT_DECAY",
        help="List of values for weight decay \
            (used when HPO is disabled).",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        metavar="PATIENCE",
        required=False,
        help="Number of epochs with no improvement \
            after which training will be stopped.",
    )

    parser.add_argument(
        "--patience_f1",
        type=int,
        default=15,
        metavar="PATIENCE_F1",
        required=False,
        help="Number of epochs with no improvement \
            after which training will be stopped.",
    )

    parser.add_argument(
        "--epochs_to_evaluate",
        type=int,
        default=1,
        metavar="EPOCHS_TO_EVALUATE",
        required=False,
        help="Number of epochs to evaluate the \
            model during training.",
    )

    parser.add_argument(
        "--epochs_to_test",
        type=int,
        default=20,
        metavar="EPOCHS_TO_TEST",
        required=False,
        help="Number of epochs to test the \
            model during training.",
    )

    return parser

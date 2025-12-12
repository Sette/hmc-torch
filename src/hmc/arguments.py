import argparse
import json


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train a Hierarchical Multi-label Classification model."
    )

    parser.add_argument(
        "--job_id",
        type=str,
        required=False,
        help="Job id for trainer job.",
    )

    # Dataset name
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=False,
        default=None,
        help="Dataset name to be used.",
    )

    parser.add_argument(
        "--use_sample",
        type=str,
        default="false",
        choices=["true", "false"],
        metavar="USE_SAMPLE",
        required=False,
        help="Enable or disable to use a sample of data (for tests). \
                Use 'true' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--save_torch_dataset",
        type=str,
        default="true",
        choices=["true", "false"],
        metavar="USE_SAMPLE",
        required=False,
        help="Enable or disable to use save torch dataset. \
                    Use 'true' to enable and 'false' to disable.",
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

    parser.add_argument(
        "--best_theshold",
        type=str,
        default="false",
        choices=["true", "false"],
        metavar="BEST_THESHOLD",
        required=False,
        help="Enable or disable to use find the best thesholds. \
                        Use 'true' to enable and 'false' to disable.",
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
            "local_constraint",
            "local_mask",
            "local_test",
        ],
        metavar="METHOD",
        required=False,
        help="Method type to use.",
    )

    parser.add_argument(
        "--focal_loss",
        type=str,
        default="false",
        choices=["true", "false"],
        metavar="FOCAL_LOSS",
        required=False,
        help="Enable or disable Focal Loss. \
            Use 'true' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--warmup",
        type=str,
        default="false",
        choices=["true", "false"],
        metavar="WARMUP",
        required=False,
        help="Enable or disable learning rate warmup. \
            Use 'true' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--n_warmup_epochs",
        type=int,
        default=50,
        required=False,
        metavar="N_WARMUP_EPOCHS",
    )

    parser.add_argument(
        "--n_warmup_epochs_increment",
        type=int,
        default=50,
        required=False,
        metavar="N_WARMUP_EPOCHS_INCREMENT",
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
        "--hpo_by_level",
        type=str,
        default="true",
        choices=["true", "false"],
        metavar="HPO_BY_LEVEL",
        required=False,
        help="Enable or disable HPO by level. \
            Use 'true' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--model_regularization",
        type=str,
        default="false",
        choices=["residual", "soft", "false"],
        metavar="MODEL_REGULARIZATION",
        required=False,
        help="Select or disable model regularization. \
            Use 'residual' or 'soft' to enable and 'false' to disable.",
    )

    parser.add_argument(
        "--results_path",
        type=str,
        default="./results/",
        metavar="RESULTS_PATH",
        required=False,
        help="Path to save results.",
    )

    parser.add_argument(
        "--early_metric",
        type=str,
        default="f1-score",
        choices=["f1-score", "accuracy", "loss"],
        metavar="EARLY_METRIC",
        required=False,
        help="Metric to use for early stopping.",
    )

    parser.add_argument(
        "--level_model_type",
        type=str,
        default="mlp",
        choices=["mlp", "attention", "gcn", "gat"],
        metavar="LEVEL_MODEL_TYPE",
        required=False,
        help="Specific model type to use at each level. Options: 'mlp' (Multi-Layer Perceptron), \
'attention' (Attention mechanism), 'gcn' (Graph Convolutional Network), 'gat' (Graph Attention Network).",
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
        type=json.loads,  # aceita JSON (ex.: '[[128,64],[256]]')
        required=False,
        metavar="HIDDEN_DIMS",
        help="List (or list of lists) of hidden neurons. "
        "Can be passed as JSON when HPO is enabled (e.g. '[[128,64],[256]]').",
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
        default=15,
        metavar="PATIENCE",
        required=False,
        help="Number of epochs with no improvement \
            after which training will be stopped.",
    )

    parser.add_argument(
        "--patience_score",
        type=int,
        default=20,
        metavar="PATIENCE_SCORE",
        required=False,
        help="Number of epochs with no improvement \
            after which training will be stopped.",
    )

    parser.add_argument(
        "--epochs_to_evaluate",
        type=int,
        default=20,
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

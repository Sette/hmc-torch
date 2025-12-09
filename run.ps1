# Configuração de variáveis de ambiente
$env:CUDA_VISIBLE_DEVICES = "0"
$env:CUDA_LAUNCH_BLOCKING = "1"

# Lista de datasets
$DATASETS = "cellcycle_GO derisi_GO eisen_GO expr_GO gasch1_GO gasch2_GO seq_GO spo_GO cellcycle_FUN derisi_FUN eisen_FUN expr_FUN gasch1_FUN gasch2_FUN seq_FUN spo_FUN"
$DATASET_PATH = Join-Path (Get-Location) "data"
$BATCH_SIZE = 64
$NON_LIN = "relu"
$DEVICE = "cpu"
$EPOCHS = 4000
$EPOCHS_TO_EVALUATE = 20
$OUTPUT_PATH = "results"
$METHOD = "local"
$SEED = 0
$DATASET_TYPE = "arff"
$HPO = "false"
$N_TRIALS = 30
$JOB_ID = "false"
$USE_SAMPLE = "false"
$SAVE_TORCH_DATASET = "false"
$MODEL_REGULARIZATION = "false"
$LEVEL_MODEL_TYPE = "mlp"
$WARMUP = "false"
$N_WARMUP_EPOCHS = 50
$N_WARMUP_EPOCHS_INCREMENT = 50
$DATASET_NAME = "seq_FUN"
$env:PYTHONPATH = "src"
$env:DATASET_PATH = $DATASET_PATH
$env:OUTPUT_PATH = $OUTPUT_PATH

# Função para exibir ajuda
function Show-Usage {
    Write-Host "Usage: script.ps1 [options]"
    Write-Host ""
    Write-Host "Available options:"
    Write-Host "  -job_id <type>           Job id  (default: $JOB_ID)"
    Write-Host "  -dataset_name <name>     Dataset name (default: $DATASET_NAME)"
    Write-Host "  -use_sample <yes/no>     Use just a sample of data  (default: $USE_SAMPLE)"
    Write-Host "  -save_torch_dataset <yes/no> Save torch dataset (default: $SAVE_TORCH_DATASET)"
    Write-Host "  -dataset_path <path>     Dataset path (default: $DATASET_PATH)"
    Write-Host "  -seed <num>              Random seed (default: $SEED)"
    Write-Host "  -dataset_type <type>     Dataset type (default: $DATASET_TYPE)"
    Write-Host "  -batch_size <num>        Batch size (default: $BATCH_SIZE)"
    Write-Host "  -lr_values <values>      Learning rates"
    Write-Host "  -dropout_values <values> Dropout rates"
    Write-Host "  -hidden_dims <values>    Hidden dimensions"
    Write-Host "  -num_layers_values <values> Number of layers"
    Write-Host "  -weight_decay_values <values> Weight decay"
    Write-Host "  -non_lin <function>      Activation function (default: $NON_LIN)"
    Write-Host "  -device <type>           Device (cuda/cpu) (default: $DEVICE)"
    Write-Host "  -epochs <num>            Number of epochs (default: $EPOCHS)"
    Write-Host "  -output_path <path>      Output path for results (default: $OUTPUT_PATH)"
    Write-Host "  -n_trials <num>          Number of HPO trials (default: $N_TRIALS)"
    Write-Host "  -method <method>         Training method (default: $METHOD)"
    Write-Host "  -hpo <true/false>        Hyperparameter optimization (default: $HPO)"
    Write-Host "  -active_levels <num>     Number of active levels"
    Write-Host "  -model_regularization <type> Model regularization (default: $MODEL_REGULARIZATION)"
    Write-Host "  -level_model_type <type>  Specific model type to use at each level. Options: 'mlp' (Multi-Layer Perceptron), \
        'attention' (Attention mechanism), 'gcn' (Graph Convolutional Network), 'gat' (Graph Attention Network). (default: $LEVEL_MODEL_TYPE)"
    Write-Host "  -warmup <true/false>     Enable learning rate warmup (default: $WARMUP)"
    Write-Host "  -n_warmup_epochs <num>   Number of warmup epochs (default: $N_WARMUP_EPOCHS)"
    Write-Host "  -n_warmup_epochs_increment <num> Increment of warmup epochs (default: $N_WARMUP_EPOCHS_INCREMENT)"
    Write-Host "  -epochs_to_evaluate <num> Number of epochs to evaluate"
    Write-Host "  -help                    Display this message and exit"
    exit 0
}

# Processamento dos argumentos
for ($i = 0; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        "-job_id" { $JOB_ID = $args[++$i] }
        "-dataset_name" { $DATASET_NAME = $args[++$i] }
        "-use_sample" { $USE_SAMPLE = $args[++$i] }
        "-save_torch_dataset" { $SAVE_TORCH_DATASET = $args[++$i] }
        "-dataset_path" { $DATASET_PATH = $args[++$i]; $env:DATASET_PATH = $DATASET_PATH }
        "-seed" { $SEED = $args[++$i] }
        "-dataset_type" { $DATASET_TYPE = $args[++$i] }
        "-batch_size" { $BATCH_SIZE = $args[++$i] }
        "-lr_values" { $LR_VALUES = $args[++$i] -split ',' }
        "-dropout_values" { $DROPOUT_VALUES = $args[++$i] -split ',' }
        "-hidden_dims" { $HIDDEN_DIMS = $args[++$i] -split ',' }
        "-num_layers_values" { $NUM_LAYERS_VALUES = $args[++$i] -split ',' }
        "-weight_decay_values" { $WEIGHT_DECAY_VALUES = $args[++$i] -split ',' }
        "-non_lin" { $NON_LIN = $args[++$i] }
        "-device" { $DEVICE = $args[++$i] }
        "-epochs" { $EPOCHS = $args[++$i] }
        "-output_path" { $OUTPUT_PATH = $args[++$i]; $env:OUTPUT_PATH = $OUTPUT_PATH }
        "-n_trials" { $N_TRIALS = $args[++$i] }
        "-method" { $METHOD = $args[++$i] }
        "-hpo" { $HPO = $args[++$i] }
        "-active_levels" { $ACTIVE_LEVELS = $args[++$i] -split ',' }
        "-epochs_to_evaluate" { $EPOCHS_TO_EVALUATE = $args[++$i] }
        "-model_regularization" { $MODEL_REGULARIZATION = $args[++$i] }
        "-level_model_type" { $LEVEL_MODEL_TYPE = $args[++$i] }
        "-warmup" { $WARMUP = $args[++$i] }
        "-n_warmup_epochs" { $N_WARMUP_EPOCHS = $args[++$i] }
        "-n_warmup_epochs_increment" { $N_WARMUP_EPOCHS_INCREMENT = $args[++$i] }
        "-help" { Show-Usage }
        default { Write-Host "Invalid option: $($args[$i])"; Show-Usage }
    }
}

# Comando base
$cmd = "python -m hmc.main " +
    "--job_id $JOB_ID " +
    "--dataset_path $DATASET_PATH " +
    "--use_sample $USE_SAMPLE " +
    "--save_torch_dataset $SAVE_TORCH_DATASET " +
    "--batch_size $BATCH_SIZE " +
    "--dataset_type $DATASET_TYPE " +
    "--non_lin $NON_LIN " +
    "--device $DEVICE " +
    "--epochs $EPOCHS " +
    "--seed $SEED " +
    "--output_path $OUTPUT_PATH " +
    "--method $METHOD " +
    "--epochs_to_evaluate $EPOCHS_TO_EVALUATE " +
    "--hpo $HPO " +
    "--warmup $WARMUP " +
    "--n_warmup_epochs $N_WARMUP_EPOCHS " +
    "--n_warmup_epochs_increment $N_WARMUP_EPOCHS_INCREMENT " +
    "--model_regularization $MODEL_REGULARIZATION " +
    "--level_model_type $LEVEL_MODEL_TYPE " +
    "--n_trials $N_TRIALS"

if ($DATASET -eq "all") {
    foreach ($dataset_local in $DATASETS -split ' ') {
        # Extração de parâmetros do config.yaml usando yq e jq
        $HIDDEN_DIMS = (yq -j ".datasets_params.$dataset_local.hidden_dims" config.yaml | jq -c .)
        $LR_VALUES = (yq ".datasets_params.$dataset_local.lr_values[]" config.yaml) -split "`n"
        $DROPOUT_VALUES = (yq ".datasets_params.$dataset_local.dropout_values[]" config.yaml) -split "`n"
        $NUM_LAYERS_VALUES = (yq ".datasets_params.$dataset_local.num_layers_values[]" config.yaml) -split "`n"
        $WEIGHT_DECAY_VALUES = (yq ".datasets_params.$dataset_local.weight_decay_values[]" config.yaml) -split "`n"

        Write-Host "Using dataset_name: $dataset_local"
        Write-Host "Using hidden dimensions: $HIDDEN_DIMS"
        
        $cmd_dataset = $cmd
        
        if ($ACTIVE_LEVELS) {
            $cmd_dataset += " --active_levels $($ACTIVE_LEVELS -join ' ')"
        }

        if (($HPO -eq "false") -and (($METHOD -eq "local") -or ($METHOD -eq "local_constraint") -or ($METHOD -eq "local_mask"))) {
            $cmd_dataset += " --lr_values $($LR_VALUES -join ' ') " +
                "--dropout_values $($DROPOUT_VALUES -join ' ') " +
                "--hidden_dims $HIDDEN_DIMS " +
                "--num_layers_values $($NUM_LAYERS_VALUES -join ' ') " +
                "--weight_decay_values $($WEIGHT_DECAY_VALUES -join ' ')"
        }

        Write-Host "Starting experiment for dataset: $dataset_local"
        $cmd_dataset += " --dataset_name $dataset_local"
        Write-Host "Running: $cmd_dataset"
        
        $process = Start-Process -FilePath "python" -ArgumentList ($cmd_dataset -replace '^python ') -NoNewWindow -PassThru
        
        try {
            $process.WaitForExit()
        }
        catch {
            $process.Kill()
        }
    }
}
else {
    Write-Host "Using specific dataset_name: $DATASET_NAME"
    
    # Extração de parâmetros do config.yaml
    $HIDDEN_DIMS = (yq -j ".datasets_params.$DATASET_NAME.hidden_dims" config.yaml | jq -c .)
    $LR_VALUES = (yq ".datasets_params.$DATASET_NAME.lr_values[]" config.yaml) -split "`n"
    $DROPOUT_VALUES = (yq ".datasets_params.$DATASET_NAME.dropout_values[]" config.yaml) -split "`n"
    $NUM_LAYERS_VALUES = (yq ".datasets_params.$DATASET_NAME.num_layers_values[]" config.yaml) -split "`n"
    $WEIGHT_DECAY_VALUES = (yq ".datasets_params.$DATASET_NAME.weight_decay_values[]" config.yaml) -split "`n"

    Write-Host "Using hidden dimensions: $HIDDEN_DIMS"

    $cmd += " --dataset_name $DATASET_NAME"

    if ($ACTIVE_LEVELS) {
        $cmd += " --active_levels $($ACTIVE_LEVELS -join ' ')"
    }

    if (($HPO -eq "false") -and (($METHOD -eq "local") -or ($METHOD -eq "local_test") -or ($METHOD -eq "local_constraint") -or ($METHOD -eq "local_mask"))) {
        $cmd += " --lr_values $($LR_VALUES -join ' ') " +
            "--dropout_values $($DROPOUT_VALUES -join ' ') " +
            "--hidden_dims $HIDDEN_DIMS " +
            "--num_layers_values $($NUM_LAYERS_VALUES -join ' ') " +
            "--weight_decay_values $($WEIGHT_DECAY_VALUES -join ' ')"
    }

    Write-Host $cmd
    
    $process = Start-Process -FilePath "python" -ArgumentList ($cmd -replace '^python ') -NoNewWindow -PassThru
    
    try {
        $process.WaitForExit()
    }
    catch {
        $process.Kill()
    }

    Write-Host "All experiments completed!"
}

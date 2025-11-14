import os

from hmc.datasets.datasets.gofun import get_dataset_paths
from hmc.utils.dataset.parser import initialize_dataset_arff_tocsv

os.environ["DATA_FOLDER"] = "./"

if __name__ == "__main__":
    datasets = get_dataset_paths(dataset_type="arff")
    dataset_name = "seq_FUN"
    data_path = os.environ["DATA_FOLDER"]
    output_path = os.path.join(data_path, "data", "HMC_data_torch")

    # datasets_name = ['cellcycle_GO', 'derisi_GO', 'eisen_GO', 'expr_GO', 'gasch1_GO',
    #                'gasch2_GO', 'seq_GO', 'spo_GO', 'cellcycle_FUN', 'derisi_FUN',
    #                'eisen_FUN', 'expr_FUN', 'gasch1_FUN', 'gasch2_FUN',
    #               'seq_FUN', 'spo_FUN']
    datasets_name = ["seq_FUN"]
    for dataset_name in datasets_name:
        data, ontology = dataset_name.split("_")
        train, val, test = initialize_dataset_arff_tocsv(
            dataset_name, datasets, output_path
        )

        train.to_csv(dataset="train")
        val.to_csv(dataset="valid")
        test.to_csv(dataset="test")

        train.to_pt(dataset="train")
        val.to_pt(dataset="valid")
        test.to_pt(dataset="test")
        print(f"{dataset_name} has been converted to csv")

# Skip the root nodes
import os

to_skip = ["root", "GO0003674", "GO0005575", "GO0008150"]


def get_dataset_paths(dataset_path="./data", dataset_type="arff"):
    # dataset_path = os.environ("DATASET_PATH")
    if dataset_type == "arff":
        datasets = {
            "enron_others": (
                False,
                os.path.join(
                    dataset_path, "HMC_data_arff/others/Enron_corr_trainvalid.arff"
                ),
                os.path.join(dataset_path, "HMC_data_arff/others/Enron_corr_test.arff"),
            ),
            "diatoms_others": (
                False,
                os.path.join(dataset_path, "HMC_data_arff/others/Diatoms_train.arff"),
                os.path.join(dataset_path, "HMC_data_arff/others/Diatoms_test.arff"),
            ),
            "imclef07a_others": (
                False,
                os.path.join(dataset_path, "HMC_data_arff/others/ImCLEF07A_Train.arff"),
                os.path.join(dataset_path, "HMC_data_arff/others/ImCLEF07A_Test.arff"),
            ),
            "imclef07d_others": (
                False,
                os.path.join(dataset_path, "HMC_data_arff/others/ImCLEF07D_Train.arff"),
                os.path.join(dataset_path, "HMC_data_arff/others/ImCLEF07D_Test.arff"),
            ),
            "cellcycle_FUN": (
                False,
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/cellcycle_FUN/cellcycle_FUN.train.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/cellcycle_FUN/cellcycle_FUN.valid.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/cellcycle_FUN/cellcycle_FUN.test.arff",
                ),
            ),
            "derisi_FUN": (
                False,
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/derisi_FUN/derisi_FUN.train.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/derisi_FUN/derisi_FUN.valid.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/derisi_FUN/derisi_FUN.test.arff",
                ),
            ),
            "eisen_FUN": (
                False,
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/eisen_FUN/eisen_FUN.train.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/eisen_FUN/eisen_FUN.valid.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/eisen_FUN/eisen_FUN.test.arff",
                ),
            ),
            "expr_FUN": (
                False,
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/expr_FUN/expr_FUN.train.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/expr_FUN/expr_FUN.valid.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/expr_FUN/expr_FUN.test.arff",
                ),
            ),
            "gasch1_FUN": (
                False,
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/gasch1_FUN/gasch1_FUN.train.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/gasch1_FUN/gasch1_FUN.valid.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/gasch1_FUN/gasch1_FUN.test.arff",
                ),
            ),
            "gasch2_FUN": (
                False,
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/gasch2_FUN/gasch2_FUN.train.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/gasch2_FUN/gasch2_FUN.valid.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/gasch2_FUN/gasch2_FUN.test.arff",
                ),
            ),
            "seq_FUN": (
                False,
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/seq_FUN/seq_FUN.train.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/seq_FUN/seq_FUN.valid.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/seq_FUN/seq_FUN.test.arff",
                ),
            ),
            "spo_FUN": (
                False,
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/spo_FUN/spo_FUN.train.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/spo_FUN/spo_FUN.valid.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_FUN/spo_FUN/spo_FUN.test.arff",
                ),
            ),
            "cellcycle_GO": (
                True,
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/cellcycle_GO/cellcycle_GO.train.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/cellcycle_GO/cellcycle_GO.valid.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/cellcycle_GO/cellcycle_GO.test.arff",
                ),
            ),
            "derisi_GO": (
                True,
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/derisi_GO/derisi_GO.train.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/derisi_GO/derisi_GO.valid.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/derisi_GO/derisi_GO.test.arff",
                ),
            ),
            "eisen_GO": (
                True,
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/eisen_GO/eisen_GO.train.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/eisen_GO/eisen_GO.valid.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/eisen_GO/eisen_GO.test.arff",
                ),
            ),
            "expr_GO": (
                True,
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/expr_GO/expr_GO.train.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/expr_GO/expr_GO.valid.arff",
                ),
                os.path.join(
                    dataset_path, "HMC_data_arff/datasets_GO/expr_GO/expr_GO.test.arff"
                ),
            ),
            "gasch1_GO": (
                True,
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/gasch1_GO/gasch1_GO.train.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/gasch1_GO/gasch1_GO.valid.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/gasch1_GO/gasch1_GO.test.arff",
                ),
            ),
            "gasch2_GO": (
                True,
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/gasch2_GO/gasch2_GO.train.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/gasch2_GO/gasch2_GO.valid.arff",
                ),
                os.path.join(
                    dataset_path,
                    "HMC_data_arff/datasets_GO/gasch2_GO/gasch2_GO.test.arff",
                ),
            ),
            "seq_GO": (
                True,
                os.path.join(
                    dataset_path, "HMC_data_arff/datasets_GO/seq_GO/seq_GO.train.arff"
                ),
                os.path.join(
                    dataset_path, "HMC_data_arff/datasets_GO/seq_GO/seq_GO.valid.arff"
                ),
                os.path.join(
                    dataset_path, "HMC_data_arff/datasets_GO/seq_GO/seq_GO.test.arff"
                ),
            ),
            "spo_GO": (
                True,
                os.path.join(
                    dataset_path, "HMC_data_arff/datasets_GO/spo_GO/spo_GO.train.arff"
                ),
                os.path.join(
                    dataset_path, "HMC_data_arff/datasets_GO/spo_GO/spo_GO.valid.arff"
                ),
                os.path.join(
                    dataset_path, "HMC_data_arff/datasets_GO/spo_GO/spo_GO.test.arff"
                ),
            ),
        }
        return datasets

    elif dataset_type == "csv":
        dataset_path = os.path.join(dataset_path, "HMC_data_csv")
        datasets = {
            "cellcycle_FUN": (
                os.path.join(dataset_path, "cellcycle_FUN", "train.csv"),
                os.path.join(dataset_path, "cellcycle_FUN", "test.csv"),
                os.path.join(dataset_path, "cellcycle_FUN", "val.csv"),
                os.path.join(dataset_path, "cellcycle_FUN", "labels.json"),
            ),
            "derisi_FUN": (
                os.path.join(dataset_path, "derisi_FUN", "train.csv"),
                os.path.join(dataset_path, "derisi_FUN", "test.csv"),
                os.path.join(dataset_path, "derisi_FUN", "val.csv"),
                os.path.join(dataset_path, "derisi_FUN", "labels.json"),
            ),
            "eisen_FUN": (
                os.path.join(dataset_path, "eisen_FUN", "train.csv"),
                os.path.join(dataset_path, "eisen_FUN", "test.csv"),
                os.path.join(dataset_path, "eisen_FUN", "val.csv"),
                os.path.join(dataset_path, "eisen_FUN", "labels.json"),
            ),
            "expr_FUN": (
                os.path.join(dataset_path, "expr_FUN", "train.csv"),
                os.path.join(dataset_path, "expr_FUN", "test.csv"),
                os.path.join(dataset_path, "expr_FUN", "val.csv"),
                os.path.join(dataset_path, "expr_FUN", "labels.json"),
            ),
            "gasch1_FUN": (
                os.path.join(dataset_path, "gasch1_FUN", "train.csv"),
                os.path.join(dataset_path, "gasch1_FUN", "test.csv"),
                os.path.join(dataset_path, "gasch1_FUN", "val.csv"),
                os.path.join(dataset_path, "gasch1_FUN", "labels.json"),
            ),
            "gasch2_FUN": (
                os.path.join(dataset_path, "gasch2_FUN", "train.csv"),
                os.path.join(dataset_path, "gasch2_FUN", "test.csv"),
                os.path.join(dataset_path, "gasch2_FUN", "val.csv"),
                os.path.join(dataset_path, "gasch2_FUN", "labels.json"),
            ),
            "seq_FUN": (
                os.path.join(dataset_path, "seq_FUN", "train.csv"),
                os.path.join(dataset_path, "seq_FUN", "test.csv"),
                os.path.join(dataset_path, "seq_FUN", "val.csv"),
                os.path.join(dataset_path, "seq_FUN", "labels.json"),
            ),
            "spo_FUN": (
                os.path.join(dataset_path, "spo_FUN", "train.csv"),
                os.path.join(dataset_path, "spo_FUN", "test.csv"),
                os.path.join(dataset_path, "spo_FUN", "val.csv"),
                os.path.join(dataset_path, "spo_FUN", "labels.json"),
            ),
            "cellcycle_GO": (
                os.path.join(dataset_path, "cellcycle_GO", "train.csv"),
                os.path.join(dataset_path, "cellcycle_GO", "test.csv"),
                os.path.join(dataset_path, "cellcycle_GO", "val.csv"),
                os.path.join(dataset_path, "cellcycle_GO", "labels.json"),
            ),
            "derisi_GO": (
                os.path.join(dataset_path, "derisi_GO", "train.csv"),
                os.path.join(dataset_path, "derisi_GO", "test.csv"),
                os.path.join(dataset_path, "derisi_GO", "val.csv"),
                os.path.join(dataset_path, "derisi_GO", "labels.json"),
            ),
            "eisen_GO": (
                os.path.join(dataset_path, "eisen_GO", "train.csv"),
                os.path.join(dataset_path, "eisen_GO", "test.csv"),
                os.path.join(dataset_path, "eisen_GO", "val.csv"),
                os.path.join(dataset_path, "eisen_GO", "labels.json"),
            ),
            "expr_GO": (
                os.path.join(dataset_path, "expr_GO", "train.csv"),
                os.path.join(dataset_path, "expr_GO", "test.csv"),
                os.path.join(dataset_path, "expr_GO", "val.csv"),
                os.path.join(dataset_path, "expr_GO", "labels.json"),
            ),
            "gasch1_GO": (
                os.path.join(dataset_path, "gasch1_GO", "train.csv"),
                os.path.join(dataset_path, "gasch1_GO", "test.csv"),
                os.path.join(dataset_path, "gasch1_GO", "val.csv"),
                os.path.join(dataset_path, "gasch1_GO", "labels.json"),
            ),
            "gasch2_GO": (
                os.path.join(dataset_path, "gasch2_GO", "train.csv"),
                os.path.join(dataset_path, "gasch2_GO", "test.csv"),
                os.path.join(dataset_path, "gasch2_GO", "val.csv"),
                os.path.join(dataset_path, "gasch2_GO", "labels.json"),
            ),
            "seq_GO": (
                os.path.join(dataset_path, "seq_GO", "train.csv"),
                os.path.join(dataset_path, "seq_GO", "test.csv"),
                os.path.join(dataset_path, "seq_GO", "val.csv"),
                os.path.join(dataset_path, "seq_GO", "labels.json"),
            ),
            "spo_GO": (
                os.path.join(dataset_path, "spo_GO", "train.csv"),
                os.path.join(dataset_path, "spo_GO", "test.csv"),
                os.path.join(dataset_path, "spo_GO", "val.csv"),
                os.path.join(dataset_path, "spo_GO", "labels.json"),
            ),
        }
        return datasets

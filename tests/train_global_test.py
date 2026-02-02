import unittest
from unittest import mock
import sys

from hmc.main import main


class TestTrainGlobalSeqFUN(unittest.TestCase):
    def test_main_global_seq_fun_args(self):
        # monta a lista de argumentos exatamente como na linha de comando
        test_argv = [
            "hmc.trainers.main",  # argv[0] é o nome do programa
            "--dataset_path",
            "./data",
            "--batch_size",
            "4",
            "--dataset_type",
            "arff",
            "--non_lin",
            "relu",
            "--device",
            "cuda",
            "--epochs",
            "2000",
            "--seed",
            "0",
            "--output_path",
            "results",
            "--method",
            "global",
            "--epochs_to_evaluate",
            "20",
            "--hpo",
            "false",
            "--dataset_name",
            "seq_FUN",
        ]

        # substitui sys.argv apenas dentro deste contexto
        with mock.patch.object(sys, "argv", test_argv):
            # se main() levantar exceção, o teste falha
            try:
                result = main()
                self.assertIn("f1score", result)
                self.assertIn("precision", result)
                self.assertIn("recall", result)
                self.assertIn("avg_precision", result)

                self.assertAlmostEqual(
                    round(result["f1score"], 4),
                    0.3347,
                    places=4,
                    msg=f"f1-score expected 0.3347, get {result['f1score']}",
                )

                self.assertAlmostEqual(
                    round(result["avg_precision"], 4),
                    0.2931,
                    places=4,
                    msg=f"avg_precision expected 0.2931, get {result['avg_precision']}",
                )

                self.assertAlmostEqual(
                    round(result["precision"], 4),
                    0.3512,
                    places=4,
                    msg=f"precision expected 0.3512, get {result['precision']}",
                )

                self.assertAlmostEqual(
                    round(result["recall"], 4),
                    0.3196,
                    places=4,
                    msg=f"recall expected 0.3196, get {result['recall']}",
                )

            except SystemExit as e:
                # argparse chama sys.exit() em erro; se código != 0, falha
                self.assertEqual(
                    e.code,
                    0,
                    msg=f"main() saiu com código {e.code}, provavelmente erro de parsing.",
                )


if __name__ == "__main__":
    unittest.main()

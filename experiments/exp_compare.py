import logging
from sklearn.datasets import load_diabetes

from config import FixedParams, NNParams, QNNParams, QSVMParams, SVMParams
from evaluation import compare, t_student
from models.model_NN import TwoLayerModel
from models.model_QNN import VariationalQuantumCircuit
from models.model_QSVM import QSVM
from models.model_SVM import SVM
from training.train import Trainer
from utils.data_loader import load_data
from utils.optuna_utils import (
    optimize_hyperparameters_QNN,
    optimize_hyperparameters_QSVM,
    optimize_hyperparameters_SVM,
)
from utils.parameters_loader import load_params, save_params, save_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_qnn_nn():
    fixed_params = FixedParams(experiment_name="HMM_Cassification_qnn_Compare_nn")

    qnn_params_dict = load_params("QNN")
    qnn_params = QNNParams(**qnn_params_dict)
    model_qnn = VariationalQuantumCircuit(
        fixed_params=fixed_params, QNN_params=qnn_params
    )

    nn_params_dict = load_params("NN")
    nn_params = NNParams(**nn_params_dict)
    model_nn = TwoLayerModel(fixed_params=fixed_params, NN_params=nn_params)

    results = compare.compare(model_qnn, model_nn, fixed_params)
    save_results(results, "qnn_vs_nn_1.json")


def compare_qsvm_svm():
    fixed_params = FixedParams(experiment_name="HMM_Cassification_qsvm_Compare_svm")

    qsvm_params_dict = load_params("QSVM")
    qsvm_params = QSVMParams(**qsvm_params_dict)
    model_qsvm = QSVM(fixed_params=fixed_params, QSVM_params=qsvm_params)

    svm_params_dict = load_params("SVM")
    svm_params = SVMParams(**svm_params_dict)
    model_svm = SVM(fixed_params=fixed_params, SVM_params=svm_params)

    results = compare.compare(model_qsvm, model_svm, fixed_params)
    save_results(results, "qsvm_vs_svm_1.json")


def find_best_qnn():

    trainer = Trainer()

    # Optimize hyperparameters
    best_params = optimize_hyperparameters_QNN(
        trainer, n_trials=trainer.fixed_params.optuna_trials
    )
    logger.info(f"Best hyperparameters: {best_params}")
    save_params(best_params, "QNN_best_params")


def find_best_qsvm():

    trainer = Trainer()

    # Optimize hyperparameters
    best_params = optimize_hyperparameters_QSVM(
        trainer, n_trials=trainer.fixed_params.optuna_trials
    )
    logger.info(f"Best hyperparameters: {best_params}")
    save_params(best_params, "QSVM_best_params")


def t_test_with_values():
    diff = [
        0.6825000047683716 - 0.9100000262260437,
        0.5724999904632568 - 0.9075000286102295,
        0.48249998688697815 - 0.875,
        0.5475000143051147 - 0.9225000143051147,
        0.6424999833106995 - 0.8899999856948853,
    ]

    t, p = t_student.compute_corrected_ttest(
        differences=diff, df=4, n_train=1600, n_test=400
    )
    save_results({"diffs": diff, "t": t, "p": p}, "qnn vs nn")


def show_data_records():
    X, y = load_data(FixedParams().dataset_file)
    for i in range(5):
        print(
            f"Dane z akcel: {X[i][:10]} \n Dane z żyro: {X[i][10:]} \n Etykieta: {y[i]} "
        )
    for i in range(1001, 1006):
        print(
            f"Dane z akcel: {X[i][:10]} \n Dane z żyro: {X[i][10:]} \n Etykieta: {y[i]} "
        )


def regresja():
    X, y = load_diabetes(return_X_y=True)
    print(X.shape, y.shape)

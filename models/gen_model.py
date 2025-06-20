import logging
import models
from config import ModelParams, FixedParams
import models.fusion_NN
import models.fusion_NN.model_QNN_fusions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gen_model(fixed_params: FixedParams, model_params: ModelParams):
    logger.info(f"Generating model:  {model_params.model_name}")
    if model_params.model_name == "TwoLayerModel":
        return models.model_NN.TwoLayerModel(fixed_params, model_params)
    elif model_params.model_name == "VariationalQuantumCircuit":
        return models.model_QNN.VariationalQuantumCircuit(fixed_params, model_params)
    elif model_params.model_name == "SVM":
        return models.model_SVM.SVM(fixed_params, model_params)
    elif model_params.model_name == "QSVM":
        return models.model_QSVM.QSVM(fixed_params, model_params)
    elif model_params.model_name == "VQC_fusion_1":
        return models.fusion_NN.model_QNN_fusions.VQC_fusion_1(
            fixed_params, model_params
        )
    else:
        raise ValueError(f"Invalid model name: {model_params.model_name}")

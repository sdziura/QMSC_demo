from sklearn.svm import SVC

from config import SVMParams, FixedParams


class SVM:
    def __init__(self, fixed_params: FixedParams, SVM_params: SVMParams):
        self.fixed_params = fixed_params
        self.model_params = SVM_params
        self.model = SVC(
            kernel=SVM_params.kernel,
            C=SVM_params.C,
            gamma=SVM_params.gamma,
            degree=SVM_params.degree,
        )

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predict(self, X):
        return self.model.predict(X=X)

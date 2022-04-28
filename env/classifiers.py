from util import *


class SVClassifier:
    def __init__(self, C, kernel, gamma):
        self.model = SVC(C=C, kernel=kernel, gamma=gamma, class_weight='balanced')
        self._x_train = None
        self._y_train = None

    def fit(self, x_train, y_train):
        self._x_train = x_train
        self._y_train = y_train
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_predict):
        return self.model.predict(x_predict)

    def score(self, x_test, y_test):
        return self.model.score(x_test, y_test)


class DNNClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.main = nn.Sequential(
            init_(nn.Linear(input_size, 64)),
            nn.ReLU(),
            init_(nn.Linear(64, 128)),
            nn.ReLU(),
            init_(nn.Linear(128, 64)),
            nn.ReLU(),
            init_(nn.Linear(64, output_size)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.main(x)
        return x

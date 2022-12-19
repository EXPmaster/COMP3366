import os
import numpy as np
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, TwoLocal, StatePreparation, RealAmplitudes
from qiskit import Aer
from qiskit.circuit import Parameter
from qiskit.algorithms.optimizers import SPSA, SLSQP, COBYLA, ADAM
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import QasmSimulator
from sklearn.preprocessing import OneHotEncoder


def callback_graph(weights, obj_func_eval):
    print(obj_func_eval)

trainset = np.loadtxt('train_toy.csv', delimiter=',')
TRAIN_DATA, TRAIN_LABELS = trainset[:, :-1], trainset[:, -1]
testset = np.loadtxt('test_toy.csv', delimiter=',')
TEST_DATA, TEST_LABELS = testset[:, :-1], testset[:, -1]
encoder = OneHotEncoder()
TRAIN_LABELS = encoder.fit_transform(np.array(TRAIN_LABELS).reshape(-1, 1)).toarray()
TEST_LABELS = encoder.fit_transform(np.array(TEST_LABELS).reshape(-1, 1)).toarray()
quantum_instance = QuantumInstance(Aer.get_backend("aer_simulator"), shots=8192)
feature_dim = 2
# feature_map = ZFeatureMap(feature_dimension=feature_dim)
feature_map = ZZFeatureMap(feature_dimension=feature_dim)
var_form = RealAmplitudes(feature_dim, reps=2)
initial_point = np.random.random(var_form.num_parameters)
vqc = VQC(
    feature_map=feature_map,
    ansatz=var_form,
    loss='cross_entropy',
    optimizer=SPSA(maxiter=100), # COBYLA(maxiter=10),
    quantum_instance=quantum_instance,
    callback=callback_graph
)
# vqc = VQC(num_qubits=16, quantum_instance=QasmSimulator())
vqc.fit(TRAIN_DATA, TRAIN_LABELS)
# print(vqc.predict(TEST_DATA))
acc = vqc.score(TEST_DATA, TEST_LABELS)
print(acc)

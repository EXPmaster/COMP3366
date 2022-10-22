from typing import List
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


def qram(data: List) -> QuantumCircuit:
    """A naive implementation of QRAM for assigment 3 Q1 (c).
    Given the data list, it prepares the state \sum_x |x>|D(x)>,
    where the |x> is the address (index) and |D(x)> is the data in the corresponding address,
    e,g, if the list is [3, 5, 1], then the state is (|0>|3> + |1>|5> + |2>|1>) / sqrt(3).

    Args:
        data: The Python List object of the data you want to search.

    Returns:
        circuit: QuantumCircuit for QRAM.
    """
    num_addr_bits = int(np.ceil(np.log2(len(data))))
    num_data_bits = int(np.ceil(np.log2(max(data))))
    address_reg = QuantumRegister(num_addr_bits)
    data_reg = QuantumRegister(num_data_bits)
    circuit = QuantumCircuit(address_reg, data_reg, name='QRAM')
    
    for id_addr in range(len(data)):
        # load address index
        id_str = format(id_addr, 'b').zfill(len(address_reg))
        for i_id_str, b_id_str in enumerate(id_str):
            if b_id_str == '0': circuit.x(address_reg[i_id_str])
        # load data
        data_str = format(data[id_addr], 'b').zfill(len(data_reg))
        for i_data_str, b_data_str in enumerate(data_str):
            if b_data_str == '1': circuit.mct(address_reg, data_reg[i_data_str])

        for i_id_str, b_id_str in enumerate(id_str):
            if b_id_str == '0': circuit.x(address_reg[i_id_str])
        circuit.barrier()
    return circuit


def oracle_c(num_data: int) -> QuantumCircuit:
    """Oracle for assignment 3 Q1 (c).
    This oracle is used for finding if the data == 5, not the index.

    Args:
        num_data: Number of qubits in data register.

    Returns:
        Oracle circuit.
    """
    data_reg = QuantumRegister(num_data)
    ancilla = QuantumRegister(1)
    circuit = QuantumCircuit(data_reg, ancilla, name='Oracle')
    
    circuit.x(data_reg[1])
    circuit.mcx(data_reg, ancilla)
    circuit.x(data_reg[1])
    return circuit
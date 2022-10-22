from qiskit import QuantumCircuit, QuantumRegister


def oracle1() -> QuantumCircuit:
    """Oracle 1 for assignment 3 Q1 (a)

    Returns:
        Oracle circuit.
    """
    data_reg = QuantumRegister(3)
    ancilla = QuantumRegister(1)
    circuit = QuantumCircuit(data_reg, ancilla, name='Oracle1')
    
    circuit.x(data_reg[1])
    circuit.mcx(data_reg, ancilla)
    circuit.x(data_reg[1])
    return circuit


def oracle2() -> QuantumCircuit:
    """Oracle 2 for assignment 3 Q1 (a)

    Returns:
        Oracle circuit.
    """
    data_reg = QuantumRegister(3)
    ancilla = QuantumRegister(1)
    circuit = QuantumCircuit(data_reg, ancilla, name='Oracle2')
    
    circuit.x(data_reg[1])
    circuit.mcx(data_reg, ancilla)
    circuit.x(data_reg[1])
    circuit.x(data_reg[0])
    circuit.mcx(data_reg, ancilla)
    circuit.x(data_reg[0])
    return circuit

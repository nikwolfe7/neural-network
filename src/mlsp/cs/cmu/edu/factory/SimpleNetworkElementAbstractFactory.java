package mlsp.cs.cmu.edu.factory;

import mlsp.cs.cmu.edu.elements.Bias;
import mlsp.cs.cmu.edu.elements.Edge;
import mlsp.cs.cmu.edu.elements.Input;
import mlsp.cs.cmu.edu.elements.LinearOutput;
import mlsp.cs.cmu.edu.elements.Neuron;
import mlsp.cs.cmu.edu.elements.Output;

public class SimpleNetworkElementAbstractFactory implements NetworkElementAbstractFactory {

	@Override
	public Input getInputElement() {
		return new Input();
	}

	@Override
	public Edge getEdgeElement() {
		return new Edge();
	}

	@Override
	public Neuron getNeuronElement() {
		return new Neuron();
	}

	@Override
	public Output getOutputElement() {
		return new Output();
	}

	@Override
	public Bias getBiasElement() {
		return new Bias();
	}

}

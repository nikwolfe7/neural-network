package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.elements.*;
import mlsp.cs.cmu.edu.dnn.util.DefaultOutput;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

public class TanhOutputFactory implements NetworkElementAbstractFactory {

	@Override
	public Input getNewInput() {
		return new Input();
	}

	@Override
	public Output getNewOutput() {
		return new TanhOutput();
	}

	@Override
	public Neuron getNewNeuron() {
		return new TanhNeuron();
	}

	@Override
	public Edge getNewEdge() {
		return new Edge();
	}

	@Override
	public Bias getNewBias() {
		return new Bias();
	}

}

package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.elements.*;
import mlsp.cs.cmu.edu.dnn.util.DefaultOutput;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

public class TanhOutputElementFactory implements NetworkElementAbstractFactory {

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
		return new MomentumEdge(-10,10,0.000001,0.25);
	}

	@Override
	public Bias getNewBias() {
		return new Bias();
	}

}

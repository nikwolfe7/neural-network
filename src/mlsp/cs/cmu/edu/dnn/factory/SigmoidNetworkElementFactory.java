package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.elements.Bias;
import mlsp.cs.cmu.edu.dnn.elements.Edge;
import mlsp.cs.cmu.edu.dnn.elements.Input;
import mlsp.cs.cmu.edu.dnn.elements.Neuron;
import mlsp.cs.cmu.edu.dnn.elements.Output;
import mlsp.cs.cmu.edu.dnn.elements.TanhNeuron;
import mlsp.cs.cmu.edu.dnn.util.DefaultOutput;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

public class SigmoidNetworkElementFactory implements NetworkElementAbstractFactory {

	@Override
	public Input getNewInput() {
		return new Input();
	}

	@Override
	public Output getNewOutput() {
		return new Output();
	}

	@Override
	public Neuron getNewNeuron() {
		return new TanhNeuron();
	}

	@Override
	public Edge getNewEdge() {
		return new Edge(-1,1,0.00125);
	}

	@Override
	public Bias getNewBias() {
		return new Bias();
	}

}

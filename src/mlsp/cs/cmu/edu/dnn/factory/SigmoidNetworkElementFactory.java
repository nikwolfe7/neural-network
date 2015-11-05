package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.elements.AdaGradEdge;
import mlsp.cs.cmu.edu.dnn.elements.Bias;
import mlsp.cs.cmu.edu.dnn.elements.Edge;
import mlsp.cs.cmu.edu.dnn.elements.Input;
import mlsp.cs.cmu.edu.dnn.elements.MomentumEdge;
import mlsp.cs.cmu.edu.dnn.elements.Neuron;
import mlsp.cs.cmu.edu.dnn.elements.Output;
import mlsp.cs.cmu.edu.dnn.elements.RPropEdge;
import mlsp.cs.cmu.edu.dnn.elements.SimpleEdge;
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
		return new Neuron();
	}

	@Override
	public Edge getNewEdge() {
	  Edge edge = new MomentumEdge(-10, 10, 0.15, 0.15);
	  return edge;
	}

	@Override
	public Bias getNewBias() {
		return new Bias();
	}

}

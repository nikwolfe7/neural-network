package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.elements.*;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

public interface NetworkElementAbstractFactory {

	public Input getNewInput();

	public Output getNewOutput();

	public Neuron getNewNeuron();

	public Edge getNewEdge();

	public Bias getNewBias();

}

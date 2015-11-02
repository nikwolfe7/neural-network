package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.elements.CrossEntropyOutput;
import mlsp.cs.cmu.edu.dnn.elements.Edge;
import mlsp.cs.cmu.edu.dnn.elements.Output;
import mlsp.cs.cmu.edu.dnn.elements.SimpleEdge;


public class CrossEntropyNetworkElementFactory extends SigmoidNetworkElementFactory {

	@Override
	public Output getNewOutput() {
		return new CrossEntropyOutput();
	}

	@Override
	public Edge getNewEdge() {
		return new SimpleEdge(-1,1,1);
	}

}

package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.elements.AdaGradEdge;
import mlsp.cs.cmu.edu.dnn.elements.CrossEntropyOutput;
import mlsp.cs.cmu.edu.dnn.elements.Edge;
import mlsp.cs.cmu.edu.dnn.elements.MomentumEdge;
import mlsp.cs.cmu.edu.dnn.elements.Output;
import mlsp.cs.cmu.edu.dnn.elements.SimpleEdge;


public class CrossEntropyNetworkElementFactory extends SigmoidNetworkElementFactory {

	@Override
	public Output getNewOutput() {
		return new CrossEntropyOutput();
	}

	@Override
	public Edge getNewEdge() {
		return new MomentumEdge(-10, 10, 0.00001, 0.1);
	}

}

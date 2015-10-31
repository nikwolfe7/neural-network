package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.elements.CrossEntropyOutput;
import mlsp.cs.cmu.edu.dnn.elements.Edge;
import mlsp.cs.cmu.edu.dnn.elements.Output;


public class CrossEntropyNetworkElementFactory extends SigmoidNetworkElementFactory {

	@Override
	public Output getNewOutput() {
		return new CrossEntropyOutput();
	}

	@Override
	public Edge getNewEdge() {
	  Edge edge = new Edge(-1,1,1);
	  edge.setAdaGrad(true);
	  edge.setMomentum(true, 0.9);
		return edge;
	}

}

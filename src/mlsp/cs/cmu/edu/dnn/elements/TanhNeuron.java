package mlsp.cs.cmu.edu.dnn.elements;

import mlsp.cs.cmu.edu.dnn.util.ActivationFunction;

public class TanhNeuron extends Neuron {

  private static final long serialVersionUID = 5932029667602181226L;

  	@Override
	public void forward() {
		double sum = 0;
	    for(NetworkElement e : getIncomingElements())
	      sum += e.getOutput();
	    setOutput(ActivationFunction.tanh(sum));
	}
	
	@Override
	public double derivative() {
		return ActivationFunction.tanhDerivative(getOutput());
	}

}

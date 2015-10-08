package mlsp.cs.cmu.edu.dnn.elements;

import mlsp.cs.cmu.edu.dnn.util.ActivationFunction;

public class TanhOutput extends Output {

	@Override
	public double derivative() {
		return ActivationFunction.tanhDerivative(getOutput());
	}

}

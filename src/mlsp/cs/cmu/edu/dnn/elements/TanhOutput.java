package mlsp.cs.cmu.edu.dnn.elements;

import mlsp.cs.cmu.edu.dnn.util.ActivationFunction;

public class TanhOutput extends Output {

  private static final long serialVersionUID = 8599918931014584940L;

  @Override
	public double derivative() {
		return ActivationFunction.tanhDerivative(getOutput());
	}

}

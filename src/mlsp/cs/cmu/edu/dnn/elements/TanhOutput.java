package mlsp.cs.cmu.edu.dnn.elements;

import mlsp.cs.cmu.edu.dnn.util.ActivationFunction;
import mlsp.cs.cmu.edu.dnn.util.CostFunction;

public class TanhOutput extends Output {

  private static final long serialVersionUID = 8599918931014584940L;

  @Override
	public double derivative() {
    double costFuncDerivative = CostFunction.meanSqErrorDerivative(getOutput(), getTruthValue());
    double tanhDerivative = ActivationFunction.tanhDerivative(getOutput());
    return costFuncDerivative * tanhDerivative;
	}

}

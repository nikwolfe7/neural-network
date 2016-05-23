package mlsp.cs.cmu.edu.dnn.elements;

import mlsp.cs.cmu.edu.dnn.util.ActivationFunction;
import mlsp.cs.cmu.edu.dnn.util.CostFunction;

/**
 * This expects to be packed into a softmax layer, mind you
 * 
 * @author Nikolas Wolfe
 *
 */
public class CrossEntropyTanhOutput extends Output {

	private static final long serialVersionUID = 7403305380238578951L;

	@Override
	public double derivative() {
		double costFuncDerivative = CostFunction.crossEntropyDerivative(getOutput(), getTruthValue());
		double tanhDerivative = ActivationFunction.tanhDerivative(getOutput());
		return costFuncDerivative * tanhDerivative;
	}

}

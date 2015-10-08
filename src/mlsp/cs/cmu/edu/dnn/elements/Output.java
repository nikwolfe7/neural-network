package mlsp.cs.cmu.edu.dnn.elements;

import mlsp.cs.cmu.edu.dnn.util.CostFunction;

/**
 * Output based on derivative of mean
 * squared error: 0.5 * (O - T) ^ 2
 * 
 * Output derivative is (O - T) 
 * 
 * @author nwolfe
 */
public class Output extends Neuron {

	private volatile double outputTruthValue = 0;

	@Override
	public void backward() {
		double mseDerivative = CostFunction.meanSqErrorDerivative(getOutput(), outputTruthValue);
		double errorTerm = mseDerivative * derivative();
		setErrorTerm(errorTerm);
	}

	public void setTruthValue(double val) {
		outputTruthValue = val;
	}
	
	public double getTruthValue() {
		return outputTruthValue;
	}

}
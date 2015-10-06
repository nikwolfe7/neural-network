package mlsp.cs.cmu.edu.elements;

import mlsp.cs.cmu.edu.util.CostFunction;

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

}

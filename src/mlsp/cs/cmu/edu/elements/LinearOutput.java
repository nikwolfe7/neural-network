package mlsp.cs.cmu.edu.elements;

import mlsp.cs.cmu.edu.util.CostFunction;

/**
 * Output based on derivative of mean
 * squared error: 0.5 * (O - T) ^ 2
 * 
 * @author nwolfe
 */
public class LinearOutput extends Output {

	private volatile double outputTruthValue = 0;

	@Override
	public void forward() {
		double sum = 0;
		for (NetworkElement e : getIncomingElements())
			sum += e.getOutput();
		setOutput(sum);
	}

	@Override
	public void backward() {
		setErrorTerm(derivative());
	}

	@Override
	public double derivative() {
		/* output derivative is (O - T) */
		return CostFunction.meanSqErrorDerivative(getOutput(), outputTruthValue);
	}

	public void setTruthValue(double val) {
		outputTruthValue = val;
	}

}

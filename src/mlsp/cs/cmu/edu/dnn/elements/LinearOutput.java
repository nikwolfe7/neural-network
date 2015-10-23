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
public class LinearOutput extends Output {

  private static final long serialVersionUID = 3008584528210916008L;

  @Override
	public void forward() {
		double sum = 0;
		for (NetworkElement e : getIncomingElements())
			sum += e.getOutput();
		setOutput(sum);
	}

	@Override
	public double derivative() {
		return CostFunction.meanSqErrorDerivative(getOutput(), getTruthValue());
	}

}

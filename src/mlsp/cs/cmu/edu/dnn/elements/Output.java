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

  private static final long serialVersionUID = -785924549621505371L;
  
  private volatile double outputTruthValue = 0;

	@Override
	public void backward() {
		double mseDerivative = CostFunction.meanSqErrorDerivative(getOutput(), outputTruthValue);
		double gradient = mseDerivative * derivative();
		setGradient(gradient);
	}

	public void setTruthValue(double val) {
		outputTruthValue = val;
	}
	
	public double getTruthValue() {
		return outputTruthValue;
	}

}

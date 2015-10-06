package mlsp.cs.cmu.edu.elements;

public class Output extends Neuron {

	private volatile double outputTruthValue = 0;

	@Override
	public void backward() {
		double errorTerm = (getOutput() - outputTruthValue) * derivative();
		setErrorTerm(errorTerm);
	}

	public void setTruthValue(double val) {
		outputTruthValue = val;
	}

}

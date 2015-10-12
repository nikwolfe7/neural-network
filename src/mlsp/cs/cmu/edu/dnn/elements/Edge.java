package mlsp.cs.cmu.edu.dnn.elements;

public class Edge implements NetworkElement {

  private static final long serialVersionUID = -3785529802453031665L;

  private double weight, output, errorTerm;

	private NetworkElement incoming, outgoing;

	/* Initializations from Tom Mitchell */
	private double initLow;
	private double initHigh; 
	private double learningRate; 

	public Edge() {
		this.initLow = -0.05;
		this.initHigh = 0.05;
		this.learningRate = 0.05;
		this.output = 0;
		this.errorTerm = 0;
		this.weight = initializeWeight(initLow, initHigh);
	}

	public Edge(double low, double high, double rate) {
		this.initLow = low;
		this.initHigh = high;
		this.learningRate = rate;
		this.output = 0;
		this.errorTerm = 0;
		this.weight = initializeWeight(initLow, initHigh);
	}

	private double initializeWeight(double low, double high) {
		return Math.random() * (high - low) + low;
	}

	public void setIncomingElement(NetworkElement element) {
		this.incoming = element;
	}

	public void setOutgoingElement(NetworkElement element) {
		this.outgoing = element;
	}

	public double getWeight() {
		return weight;
	}

	@Override
	public void forward() {
		output = weight * incoming.getOutput();
	}

	@Override
	public void backward() {
		errorTerm = outgoing.getErrorTerm() * derivative();
		double update = weight - learningRate * errorTerm;
		weight = update;
	}

	@Override
	public double derivative() {
		/* w.r.t. the weights, (w * x), derivative is x */
		return incoming.getOutput();
	}

	@Override
	public double getOutput() {
		return output;
	}

	@Override
	public double getErrorTerm() {
		return errorTerm;
	}

}

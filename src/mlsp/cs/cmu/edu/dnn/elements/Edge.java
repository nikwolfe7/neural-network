package mlsp.cs.cmu.edu.dnn.elements;

public class Edge implements NetworkElement {

	private static final long serialVersionUID = -3785529802453031665L;

	private boolean batchUpdate;

	private double weight, output, gradient, batchSum;

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
		this.gradient = 0;
		this.batchSum = 0;
		this.batchUpdate = false;
		this.weight = initializeWeight(initLow, initHigh);
	}

	public Edge(double low, double high, double rate) {
		this.initLow = low;
		this.initHigh = high;
		this.learningRate = rate;
		this.output = 0;
		this.gradient = 0;
		this.batchSum = 0;
		this.batchUpdate = false;
		this.weight = initializeWeight(initLow, initHigh);
	}

	public Edge(boolean batch, double low, double high, double rate) {
		this.initLow = low;
		this.initHigh = high;
		this.learningRate = rate;
		this.output = 0;
		this.gradient = 0;
		this.batchSum = 0;
		this.batchUpdate = batch;
		this.weight = initializeWeight(initLow, initHigh);
	}

	public void reinitializeWeight(double low, double high) {
		weight = initializeWeight(low, high);
	}

	private double initializeWeight(double low, double high) {
		return Math.random() * (high - low) + low;
	}

	public void setLearningRate(double rate) {
		learningRate = rate;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setIncomingElement(NetworkElement element) {
		this.incoming = element;
	}

	public void setOutgoingElement(NetworkElement element) {
		this.outgoing = element;
	}

	public void setBatchUpdate(boolean b) {
		this.batchUpdate = b;
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
		gradient = outgoing.getGradient() * derivative();
		updateWeight();
	}

	private void updateWeight() {
		if (batchUpdate)
			batchSum += gradient;
		else
			weight = weight - (learningRate * gradient);
	}

	public void batchUpdate() {
		if (batchUpdate) {
			weight = weight - (learningRate * batchSum);
			batchSum = 0;
		}
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
	public double getGradient() {
		return gradient;
	}

}

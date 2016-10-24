package mlsp.cs.cmu.edu.dnn.elements;

import mlsp.cs.cmu.edu.dnn.util.LayerElementUtils;

public abstract class Edge implements NetworkElement, Cloneable, Freezable {

	private static final long serialVersionUID = -3785529802453031665L;

	/* Freezable Weight */
	private boolean frozen = false;
	
	/* Batch Training */
	private boolean batchUpdate = false;
	private double batchSum = 0;

	/* Edge parameters */
	private double weight, output, gradient;

	/* Initializations from Tom Mitchell */
	private double learningRate;

	private NetworkElement incoming, outgoing;

	public Edge() {
		this.learningRate = 0.05;
		this.output = 0;
		this.gradient = 0;
		this.batchSum = 0;
		this.weight = initializeWeight(-0.05, 0.05);
	}

	public Edge(double low, double high, double rate) {
		this.learningRate = rate;
		this.output = 0;
		this.gradient = 0;
		this.batchSum = 0;
		this.weight = initializeWeight(low, high);
	}

	public void setBatchUpdate(boolean b) {
		batchUpdate = b;
		batchSum = 0;
	}

	public boolean isBatchUpdate() {
		return batchUpdate;
	}

	public void resetBatchGradient() {
		batchSum = 0;
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
		incoming = element;
	}

	public NetworkElement getIncomingElement() {
		return incoming;
	}

	public void setOutgoingElement(NetworkElement element) {
		outgoing = element;
	}

	public NetworkElement getOutgoingElement() {
		return outgoing;
	}

	public double getWeight() {
		return weight;
	}

	public void setWeight(double w) {
		if (!isFrozen()) weight = w;
	}

	@Override
	public void forward() {
		output = weight * incoming.getOutput();
	}

	@Override
	public void backward() {
		setGradient(outgoing.getGradient() * derivative());
		updateWeight();
	}

	public abstract void updateWeight();

	public abstract void batchUpdate();

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

	public double getBatchGradient() {
		return batchSum;
	}

	public void setGradient(double g) {
		batchSum += g;
		gradient = g;
	}

	@Override
	public Object clone() {
		Edge o = null;
		try {
			o = (Edge) super.clone();
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
		LayerElementUtils.convertEdge(this, o);
		return o;
	}

	@Override
	public void setFrozen(boolean b) {
		frozen = b;
	}

	@Override
	public boolean isFrozen() {
		return frozen;
	}

}

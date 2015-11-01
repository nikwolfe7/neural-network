package mlsp.cs.cmu.edu.dnn.elements;

public class SimpleEdge extends Edge {

	private static final long serialVersionUID = -5379408040978294540L;

	public SimpleEdge() {
		super();
	}

	public SimpleEdge(double low, double high, double rate) {
		super(low, high, rate);
	}

	@Override
	public void updateWeight() {
		if (!isBatchUpdate()) {
			double update = getLearningRate() * getGradient();
			double w = getWeight() - update;
			setWeight(w);
		}
	}

	@Override
	public void batchUpdate() {
		double update = getLearningRate() * getBatchGradient();
		double w = getWeight() - update;
		setWeight(w);
		resetBatchGradient();
	}

}

package mlsp.cs.cmu.edu.dnn.elements;

public class MomentumEdge extends Edge {

	private static final long serialVersionUID = 4291621900344745625L;

	private double alpha = 0;
	private double prevUpdate = 0;

	public MomentumEdge(double alpha) {
		super();
		this.alpha = alpha;
	}

	public MomentumEdge(double low, double high, double rate, double alpha) {
		super(low, high, rate);
		this.alpha = alpha;
	}

	@Override
	public void updateWeight() {
		if (!isBatchUpdate()) {
			double momentum = alpha * prevUpdate;
			double update = getLearningRate() * getGradient();
			update += momentum;
			prevUpdate = update;
			double w = getWeight() - update;
			setWeight(w);
		}
	}

	@Override
	public void batchUpdate() {
		double momentum = alpha * prevUpdate;
		double update = getLearningRate() * getBatchGradient();
		update += momentum;
		prevUpdate = update;
		double w = getWeight() - update;
		setWeight(w);
	}
}

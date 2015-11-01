package mlsp.cs.cmu.edu.dnn.elements;

public class AdaGradEdge extends Edge {

	private static final long serialVersionUID = 3275599854951059967L;
	private double adaGradientSum = Double.MIN_VALUE;

	public AdaGradEdge() {
		super();
	}

	public AdaGradEdge(double low, double high, double rate) {
		super(low, high, rate);
	}

	@Override
	public void updateWeight() {
		double gradient = getGradient();
		adaGradientSum += Math.pow(gradient, 2);
		if (!isBatchUpdate()) {
			double update = (newLearningRate(gradient) * gradient);
			double w = getWeight() - update;
			setWeight(w);
		}
	}

	@Override
	public void batchUpdate() {
		double batchGradient = getBatchGradient();
		double update = (newLearningRate(batchGradient) * batchGradient);
		double w = getWeight() - update;
		setWeight(w);
		resetBatchGradient();
	}

	private double newLearningRate(double gradient) {
		double denom = Math.sqrt(adaGradientSum);
		double newLearningRate = getLearningRate() / denom;
		return newLearningRate;
	}

}

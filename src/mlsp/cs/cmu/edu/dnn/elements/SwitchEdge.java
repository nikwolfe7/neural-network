package mlsp.cs.cmu.edu.dnn.elements;

public class SwitchEdge extends Edge implements Switchable, SecondDerivativeNetworkElement {

	private static final long serialVersionUID = 1338100192118401076L;

	private Edge edge;
	private boolean switchOff;
	private double secondGradient;

	public SwitchEdge(Edge e) {
		this.edge = e;
		this.switchOff = false;
		this.secondGradient = 0;
	}

	@Override
	public void setSwitchOff(boolean b) {
		switchOff = b;
	}

	@Override
	public void backward() {
		setGradient(getOutgoingElement().getGradient() * derivative());
		SecondDerivativeNetworkElement elem = (SecondDerivativeNetworkElement) getOutgoingElement();
		setSecondGradient(elem.getSecondGradient() * Math.pow(getWeight(),2));
		updateWeight();
	}

	private void setSecondGradient(double d) {
	  secondGradient = d;
  }

  @Override
	public void updateWeight() {
		if (!switchOff)
			edge.updateWeight();
	}

	@Override
	public void batchUpdate() {
		if (!switchOff)
			edge.batchUpdate();
	}

	@Override
	public void setBatchUpdate(boolean b) {
		edge.setBatchUpdate(b);
	}

	@Override
	public boolean isBatchUpdate() {
		return edge.isBatchUpdate();
	}

	@Override
	public void resetBatchGradient() {
		edge.resetBatchGradient();
	}

	@Override
	public void reinitializeWeight(double low, double high) {
		edge.reinitializeWeight(low, high);
	}

	@Override
	public void setLearningRate(double rate) {
		edge.setLearningRate(rate);
	}

	@Override
	public double getLearningRate() {
		return edge.getLearningRate();
	}

	@Override
	public void setIncomingElement(NetworkElement element) {
		edge.setIncomingElement(element);
	}

	@Override
	public NetworkElement getIncomingElement() {
		return edge.getIncomingElement();
	}

	@Override
	public void setOutgoingElement(NetworkElement element) {
		edge.setOutgoingElement(element);
	}

	@Override
	public NetworkElement getOutgoingElement() {
		return edge.getOutgoingElement();
	}

	@Override
	public double getWeight() {
		return edge.getWeight();
	}

	@Override
	public void setWeight(double w) {
		edge.setWeight(w);
	}

	@Override
	public void forward() {
		edge.forward();
	}

	@Override
	public double derivative() {
		return edge.derivative();
	}

	@Override
	public double getOutput() {
		return edge.getOutput();
	}

	@Override
	public double getGradient() {
		return edge.getGradient();
	}

	@Override
	public double getBatchGradient() {
		return edge.getBatchGradient();
	}

	@Override
	public void setGradient(double g) {
		edge.setGradient(g);
	}

  @Override
  public double secondDerivative() {
    return 1;
  }

  @Override
  public double getSecondGradient() {
     return secondGradient;
  }

}

package mlsp.cs.cmu.edu.dnn.elements;

public class SwitchEdge extends Edge implements Switchable {

	private static final long serialVersionUID = 1338100192118401076L;

	private Edge edge;
	private boolean switchOff;

	public SwitchEdge(Edge e) {
		this.edge = e;
		this.switchOff = false;
	}

	@Override
	public void setSwitchOff(boolean b) {
		switchOff = b;
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
	public void setBatchUpdate(boolean b) {
		edge.setBatchUpdate(b);
	}
	
	@Override
	public double getWeight() {
		return edge.getWeight();
	}
	
	@Override
	public void forward() {
		edge.forward();
	}
	
	@Override
	public void backward() {
		setGradient(getOutgoingElement().getGradient() * edge.derivative());
		updateWeight();
	}
	
	@Override
	protected void updateWeight() {
		if (!switchOff)
			edge.updateWeight();
	}

	@Override
	public void batchUpdate() {
		if (!switchOff)
			edge.batchUpdate();
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
	public void setGradient(double g) {
		edge.setGradient(g);
	}

}

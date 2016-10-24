package mlsp.cs.cmu.edu.dnn.elements;

import mlsp.cs.cmu.edu.dnn.util.LayerElementUtils;

public class SwitchEdge extends SimpleEdge implements Switchable, SecondDerivativeNetworkElement {

	private static final long serialVersionUID = 1338100192118401076L;

	private boolean switchOff;
	private double secondGradient;

	public SwitchEdge(Edge e) {
		LayerElementUtils.convertEdge(e, this);
		this.switchOff = false;
		this.secondGradient = 0;
	}

	public void reset() {
		setSecondGradient(0);
		setGradient(0);
		resetBatchGradient();
	}

	@Override
	public void setSwitchedOff(boolean b) {
		switchOff = b;
	}

	@Override
	public boolean isSwitchedOff() {
		return switchOff;
	}

	@Override
	public void backward() {
		setGradient(getOutgoingElement().getGradient() * derivative());
		SecondDerivativeNetworkElement elem = (SecondDerivativeNetworkElement) getOutgoingElement();
		setSecondGradient(elem.getSecondGradient() * Math.pow(getWeight(), 2));
		updateWeight();
	}

	private void setSecondGradient(double d) {
		secondGradient = d;
	}

	@Override
	public void updateWeight() {
		if (!switchOff)
			super.updateWeight();
	}

	@Override
	public void batchUpdate() {
		if (!switchOff)
			super.batchUpdate();
	}

	@Override
	public double secondDerivative() {
		return 1.0;
	}

	@Override
	public double getSecondGradient() {
		return secondGradient;
	}

}

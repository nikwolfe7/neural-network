package mlsp.cs.cmu.edu.dnn.elements;

public class GainSwitchNeuron implements NetworkElement {

	private static final long serialVersionUID = 8508124798661568974L;
	
	private Neuron neuron;
	private boolean switchedOff;
	private double gain;
	
	public GainSwitchNeuron(Neuron n) {
		this.neuron = n;
		this.switchedOff = false;
		this.gain = 0;
	}
	
	public void clearGain() {
		this.gain = 0;
	}
	
	public double getGain() {
		return gain;
	}
	
	public void switchOff(boolean b) {
		switchedOff = b;
	}

	@Override
	public void forward() {
		if(!switchedOff)
			neuron.forward();
	}

	@Override
	public void backward() {
		if(!switchedOff)
			neuron.forward();
	}

	@Override
	public double derivative() {
		if(!switchedOff)
			return neuron.derivative();
		else
			return 0;
	}

	@Override
	public double getOutput() {
		if(!switchedOff)
			return neuron.getOutput();
		else
			return 0;
	}

	@Override
	public double getGradient() {
		if(!switchedOff)
			return neuron.getGradient();
		else
			return 0;
	}

}

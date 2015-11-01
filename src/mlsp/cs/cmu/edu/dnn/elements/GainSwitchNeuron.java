package mlsp.cs.cmu.edu.dnn.elements;

import java.util.List;

public class GainSwitchNeuron extends Neuron implements Switchable {

	private static final long serialVersionUID = 6087613637614963686L;
	
	private static int IDNumber = 0;

	private Neuron neuron;

	private boolean switchOff;

	private double gainSum;
	
	private double groundTruthError;
	
	private int count;
	
	private int idNum;

	public GainSwitchNeuron(Neuron n) {
	  GainSwitchNeuron.IDNumber++;
	  this.idNum = IDNumber; 
		this.neuron = n;
		this.switchOff = false;
		this.gainSum = 0;
		this.count = 0;
	}

	@Override
	public void setSwitchOff(boolean b) {
		switchOff = b;
		setOutput(0);
		setGradient(0);
	}

	public void reset() {
		count = 0;
		gainSum = 0;
	}
	
	public int getIdNum() {
	  return idNum;
	}
	
	public void setGroundTruthError(double val) {
	  groundTruthError = val;
	}
	
	public double getGroundTruthError() {
	  return groundTruthError;
	}

	public double getTotalGain() {
		return gainSum;
	}

	public double getAverageGain() {
		return getTotalGain() / count;
	}
	
	@Override
	public void addIncomingElement(NetworkElement element) {
		neuron.addIncomingElement(element);
	}

	@Override
	public void addOutgoingElement(NetworkElement element) {
		neuron.addOutgoingElement(element);
	}

	@Override
	public void forward() {
		if (!switchOff) {
			neuron.forward();
		}
	}

	@Override
	public void backward() {
		if (!switchOff) {
			neuron.backward();
			count++;
			gainSum += (getGradient() * getOutput());
		}
	}
	
	@Override
	public void setGradient(double e) {
		if (!switchOff)
			neuron.setGradient(e);
	}

	@Override
	public double derivative() {
		if (!switchOff)
			return neuron.derivative();
		else
			return 0;
	}

	@Override
	public double getOutput() {
		if (!switchOff)
			return neuron.getOutput();
		else
			return 0;
	}

	@Override
	public double getGradient() {
		if (!switchOff)
			return neuron.getGradient();
		else
			return 0;
	}

	@Override
	public List<NetworkElement> getIncomingElements() {
		return neuron.getIncomingElements();
	}

	@Override
	public List<NetworkElement> getOutgoingElements() {
		return neuron.getOutgoingElements();
	}
	
	@Override
	public void setOutput(double output) {
		neuron.setOutput(output);
	}

}

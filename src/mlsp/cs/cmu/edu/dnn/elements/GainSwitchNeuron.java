package mlsp.cs.cmu.edu.dnn.elements;

import mlsp.cs.cmu.edu.dnn.util.ActivationFunction;
import mlsp.cs.cmu.edu.dnn.util.LayerElementUtils;

public class GainSwitchNeuron extends Neuron implements Switchable, SecondDerivativeNetworkElement {

	private static final long serialVersionUID = 6087613637614963686L;

	private static int IDNumber = 0;

	private boolean switchedOff;
	
	private double gain;

	private double gradientSum;

	private double secondGainSum;

	private double groundTruthError;

	private double secondGradient;

	private int count;

	private int idNum;

	public GainSwitchNeuron(Neuron n) {
		LayerElementUtils.convertNeuron(n, this);
		GainSwitchNeuron.IDNumber++;
		this.idNum = IDNumber;
		this.switchedOff = false;
		this.gain = 1.0;
		this.gradientSum = 0;
		this.secondGainSum = 0;
		this.count = 0;
		this.secondGradient = 0;
	}

	@Override
	public void setSwitchOff(boolean b) {
		switchedOff = b;
		setOutput(0);
	}
	
	@Override
	public boolean isSwitchedOff() {
		return switchedOff;
	}
	
	public double getGain() {
	  return gain;
	}
	
	public void setGain(double d) {
	  gain = d;
	}

	public void reset() {
		count = 0;
		gain = 1.0;
		gradientSum = 0;
		secondGainSum = 0;
		setGradient(0);
		setSecondGradient(0);
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
		return -1.0 * gradientSum;
	}

	public double getTotalSecondGain() {
		return secondGainSum * 0.5 - gradientSum;
	}

	public double getAverageGain() {
		return getTotalGain() / count;
	}

	public double getAverageSecondGain() {
		return getTotalSecondGain() / count;
	}

	@Override
	public void forward() {
		if (!switchedOff) {
			super.forward();
		}
	}

	@Override
	public void backward() {
		if (!switchedOff) {
			super.backward();
			double secondGradSum = 0;
			for (NetworkElement e : getOutgoingElements()) {
				SecondDerivativeNetworkElement elem = (SecondDerivativeNetworkElement) e;
				secondGradSum += elem.getSecondGradient();
			}
			double secondGrad = secondGradSum * Math.pow(derivative(), 2) + getGradient() * secondDerivative();
			setSecondGradient(secondGrad);
			count++;
			gradientSum += (getGradient() * getOutput());
			secondGainSum += (secondGradSum * Math.pow(getOutput(), 2));
		}
	}

	private void setSecondGradient(double val) {
		secondGradient = val;
	}

	@Override
	public void setGradient(double e) {
		if (!switchedOff)
			super.setGradient(e);
	}

	@Override
	public double derivative() {
		if (!switchedOff)
			return super.derivative();
		else
			return 0;
	}

	@Override
	public double getOutput() {
		if (!switchedOff)
		   // this is where the gain gets used...
			return super.getOutput() * gain;
		else
			return 0;
	}

	@Override
	public double getGradient() {
		if (!switchedOff)
			return super.getGradient();
		else
			return 0;
	}

	@Override
	public double secondDerivative() {
		return ActivationFunction.sigmoidSecondDerivative(getOutput());
	}

	@Override
	public double getSecondGradient() {
		return secondGradient;
	}

}

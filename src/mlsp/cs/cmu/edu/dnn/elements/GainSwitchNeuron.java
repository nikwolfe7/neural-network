package mlsp.cs.cmu.edu.dnn.elements;

import mlsp.cs.cmu.edu.dnn.structure.Switchable;

public class GainSwitchNeuron extends Neuron implements Switchable {

  private static final long serialVersionUID = 6087613637614963686L;

  private Neuron neuron;

  private boolean switchOff;
  
  private double gainSum;
  private double gradientSum;
  private int count;

  public GainSwitchNeuron(Neuron n) {
    this.neuron = n;
    this.switchOff = false;
    this.gainSum = 0;
    this.gradientSum = 0;
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
    gradientSum = 0;
  }
  
  public double getTotalGain() {
    return gainSum * gradientSum;
  }
  
  public double getAverageGain() {
    return getTotalGain() / count;
  }

  @Override
  public void forward() {
    if (!switchOff) {
      neuron.forward();
      gainSum += getOutput();
      count++;
    }
  }

  @Override
  public void backward() {
    if (!switchOff) {
      neuron.backward();
      gradientSum += getGradient();
    }
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

}

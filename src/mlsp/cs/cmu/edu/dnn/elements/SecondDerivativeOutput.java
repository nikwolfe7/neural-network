package mlsp.cs.cmu.edu.dnn.elements;

import java.util.List;

import mlsp.cs.cmu.edu.dnn.util.ActivationFunction;

public class SecondDerivativeOutput extends Output implements SecondDerivativeNetworkElement {

  private static final long serialVersionUID = -887267539965711250L;
  
  private double secondGradient;
  
  private Output output;
  
  public SecondDerivativeOutput(Output output) {
     this.output = output;
  }
  
  @Override
  public void backward() {
    output.backward();
    setSecondGradient(secondDerivative());
  }

  private void setSecondGradient(double val) {
    secondGradient = val;
  }

  @Override
  public double secondDerivative() {
    return ActivationFunction.sigmoidSecondDerivative(getOutput());
  }

  @Override
  public double getSecondGradient() {
    return secondGradient;
  }

  @Override
  public double derivative() {
    return output.derivative();
  }

  @Override
  public void setTruthValue(double val) {
    output.setTruthValue(val);
  }

  @Override
  public double getTruthValue() {
    return output.getTruthValue();
  }

  @Override
  public void addIncomingElement(NetworkElement element) {
    output.addIncomingElement(element);
  }

  @Override
  public void addOutgoingElement(NetworkElement element) {
    output.addOutgoingElement(element);
  }

  @Override
  public void forward() {
    output.forward();
  }

  @Override
  public void setGradient(double e) {
    output.setGradient(e);
  }

  @Override
  public double getOutput() {
    return output.getOutput();
  }

  @Override
  public double getGradient() {
    return output.getGradient();
  }

  @Override
  public List<NetworkElement> getIncomingElements() {
    return output.getIncomingElements();
  }

  @Override
  public List<NetworkElement> getOutgoingElements() {
    return output.getOutgoingElements();
  }

  @Override
  public void setOutput(double o) {
    output.setOutput(o);
  }

}

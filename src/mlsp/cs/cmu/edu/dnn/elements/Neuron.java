package mlsp.cs.cmu.edu.dnn.elements;

import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.util.ActivationFunction;

public class Neuron implements NetworkElement {

  private static final long serialVersionUID = 1621097065271638526L;
  
  private List<NetworkElement> incoming;
  private List<NetworkElement> outgoing;
  private double output;
  private double gradient;
  
  public Neuron() {
    this.output = 0;
    this.gradient = 0;
    this.incoming = new ArrayList<NetworkElement>();
    this.outgoing = new ArrayList<NetworkElement>();
  }

  public void addIncomingElement(NetworkElement element) {
    incoming.add(element);
  }
  
  public void addOutgoingElement(NetworkElement element) {
    outgoing.add(element);
  }
  
  @Override
  public void forward() {
    double sum = 0;
    for(NetworkElement e : getIncomingElements())
      sum += e.getOutput();
    setOutput(ActivationFunction.sigmoid(sum));
  }

  @Override
  public void backward() {
    double sum = 0;
    for(NetworkElement e : outgoing)
      sum += e.getGradient();
    setGradient(sum * derivative());
  }
  
  /* This is used in subclasses... */
  public void setGradient(double e) {
    gradient = e;
  }

  @Override
  public double derivative() {
    return ActivationFunction.sigmoidDerivative(getOutput());
  }

  @Override
  public double getOutput() {
    return output;
  }

  @Override
  public double getGradient() {
    return gradient;
  }

  public List<NetworkElement> getIncomingElements() {
    return incoming;
  }
  
  public List<NetworkElement> getOutgoingElements() {
    return outgoing;
  }

  public void setOutput(double output) {
    this.output = output;
  }

  @Override
  public void remove() {
    for(NetworkElement e : getIncomingElements())
      e.remove(this);
    for(NetworkElement e : getOutgoingElements())
      e.remove(this);
  }

  @Override
  public void remove(NetworkElement e) {
    getIncomingElements().remove(e);
    getOutgoingElements().remove(e);
  }

}

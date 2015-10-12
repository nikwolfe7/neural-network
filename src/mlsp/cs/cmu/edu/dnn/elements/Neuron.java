package mlsp.cs.cmu.edu.dnn.elements;

import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.util.ActivationFunction;

public class Neuron implements NetworkElement {

  private static final long serialVersionUID = 1621097065271638526L;
  
  private List<NetworkElement> incoming;
  private List<NetworkElement> outgoing;
  private double output;
  private double errorTerm;
  
  public Neuron() {
    this.output = 0;
    this.errorTerm = 0;
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
      sum += e.getErrorTerm();
    setErrorTerm(sum * derivative());
  }
  
  /* This is used in subclasses... */
  protected void setErrorTerm(double e) {
    errorTerm = e;
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
  public double getErrorTerm() {
    return errorTerm;
  }

  protected List<NetworkElement> getIncomingElements() {
    return incoming;
  }

  protected void setOutput(double output) {
    this.output = output;
  }

}

package mlsp.cs.cmu.edu.elements;

import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.util.ActivationFunction;

public class Neuron implements NetworkElement {

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
    for(NetworkElement e : incoming)
      sum += e.getOutput();
    output = ActivationFunction.getActivation(sum);
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
    return ActivationFunction.getActivationDerivative(getOutput());
  }

  @Override
  public double getOutput() {
    return output;
  }

  @Override
  public double getErrorTerm() {
    return errorTerm;
  }

}

package mlsp.cs.cmu.edu.elements;

public class Edge implements NetworkElement {

  private double weight, output, errorTerm;

  private NetworkElement incoming, outgoing;

  /* Initializations from Tom Mitchell */
  private double initLow = -5;
  private double initHigh = 5;
  private double learningRate = 0.1;

  public Edge() {
    this.output = 0;
    this.errorTerm = 0;
    this.weight = initializeWeight(initHigh, initLow);
  }
  
  private double initializeWeight(double high, double low) {
    return Math.random() * (high - low) + low;
  }
  
  public void setIncomingElement(NetworkElement element) {
    this.incoming = element;
  }

  public void setOutgoingElement(NetworkElement element) {
    this.outgoing = element;
  }
  
  public double getWeight() {
    return weight;
  }

  @Override
  public void forward() {
    output = weight * incoming.getOutput();
  }

  @Override
  public void backward() {
    errorTerm = outgoing.getErrorTerm() * derivative();
    double update = weight - learningRate * errorTerm;
    weight = update;
  }

  @Override
  public double derivative() {
    /* w.r.t. the weights, (w * x), derivative is x */
    return incoming.getOutput();
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

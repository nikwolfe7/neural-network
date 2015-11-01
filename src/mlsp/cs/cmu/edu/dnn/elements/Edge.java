package mlsp.cs.cmu.edu.dnn.elements;

public abstract class Edge implements NetworkElement {

  private static final long serialVersionUID = -3785529802453031665L;
  
  /* Optimization 2: R-Prop */
  private boolean rProp = false;
  private double nPlus = 1.2;
  private double nMinus = -0.5;
  private int prevSign = 1;

  /* Optimization 4: Batch Training */
  private boolean batchUpdate = false;
  private double batchSum = 0;
  
  /* Edge parameters */
  private double weight, output, gradient;

  /* Initializations from Tom Mitchell */
  private double initLow;
  private double initHigh;
  private double learningRate;
  
  private NetworkElement incoming, outgoing;

  public Edge() {
    this.initLow = -0.05;
    this.initHigh = 0.05;
    this.learningRate = 0.05;
    this.output = 0;
    this.gradient = 0;
    this.batchSum = 0;
    this.weight = initializeWeight(initLow, initHigh);
  }

  public Edge(double low, double high, double rate) {
    this.initLow = low;
    this.initHigh = high;
    this.learningRate = rate;
    this.output = 0;
    this.gradient = 0;
    this.batchSum = 0;
    this.weight = initializeWeight(initLow, initHigh);
  }

  
  
//  public void setRProp(boolean b) {
//    this.rProp = b;
//    this.adaGrad = !b;
//    this.prevSign = 1;
//  }
  
  public void setBatchUpdate(boolean b) {
    batchUpdate = b;
    batchSum = 0;
  }
  
  public boolean isBatchUpdate() {
	  return batchUpdate;
  }
  
  public void resetBatchGradient() {
	  batchSum = 0;
  }
  
  public void reinitializeWeight(double low, double high) {
    weight = initializeWeight(low, high);
  }

  private double initializeWeight(double low, double high) {
    return Math.random() * (high - low) + low;
  }

  public void setLearningRate(double rate) {
    learningRate = rate;
  }

  public double getLearningRate() {
    return learningRate;
  }

  public void setIncomingElement(NetworkElement element) {
    incoming = element;
  }

  public NetworkElement getIncomingElement() {
    return incoming;
  }

  public void setOutgoingElement(NetworkElement element) {
    outgoing = element;
  }

  public NetworkElement getOutgoingElement() {
    return outgoing;
  }

  public double getWeight() {
    return weight;
  }
  
  public void setWeight(double w) {
	  weight = w;
  }
  
  @Override
  public void forward() {
    output = weight * incoming.getOutput();
  }

  @Override
  public void backward() {
    setGradient(outgoing.getGradient() * derivative());
    updateWeight();
  }

  public abstract void updateWeight(); {
      weight = weight - getUpdate();
    prevSign = ((gradient >= 0) ? 1 : -1);
  }
  
/**
   * Adagrad is for stochastic grad descent. We don't use it here
   */
  public abstract void batchUpdate(); {
    if (batchUpdate) {
      double update = (learningRate * batchSum);
      if(rProp) 
        update = rPropUpdate(update);
      if(momentum)
        update = addMomentum(update);
      weight = weight - update;
      batchSum = 0;
    }
  }
  
  private double getUpdate() {
    double update;
    if(rProp) {
      update = rPropUpdate(learningRate * gradient);
    }
    if(momentum)
      update = addMomentum(update);
    return update;
  }
  
  private double rPropUpdate(double update) {
    int sign = ((gradient >= 0) ? 1 : -1);
    return (sign != prevSign) ? (update * nMinus) : (update * nPlus);
  }
  
  private double addMomentum(double update) {
    double momentum = alpha * prevUpdate;
    update = update + momentum;
    prevUpdate = update;
    return update;
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
  public double getGradient() {
    return gradient;
  }
  
  public double getBatchGradient() {
		return batchSum;
	}

	public void setGradient(double g) {
		batchSum += g;
		gradient = g;
	}
}

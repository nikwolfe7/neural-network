package mlsp.cs.cmu.edu.dnn.elements;

public class Edge implements NetworkElement {

  private static final long serialVersionUID = -3785529802453031665L;
  
  /* Optimization 1: AdaGrad */
  private boolean adaGrad = false;
  private double eps = Double.MIN_VALUE;
  private double adaGradientSum = eps;
  
  /* Optimization 2: R-Prop */
  private boolean rProp = false;
  private double nPlus = 1.2;
  private double nMinus = -0.5;
  private int prevSign = 1;
  
  /* Optimization 3: Momentum */
  private boolean momentum = false;
  private double alpha = 0;
  private double prevUpdate = 0;

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
    this.output = 0;
    this.gradient = 0;
    this.batchSum = 0;
    this.weight = initializeWeight(initLow, initHigh);
    setLearningRate(0.05);
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

  public void setMomentum(boolean b, double alpha) {
    this.momentum = b;
    this.alpha = alpha;
    this.prevUpdate = 0;
    this.rProp = (momentum) ? false : rProp;
  }
  
  public void setAdaGrad(boolean b) {
    this.adaGrad = b;
    this.adaGradientSum = eps;
    this.rProp = (adaGrad) ? false : rProp;
  }
  
  public void setRProp(boolean b) {
    this.rProp = b;
    this.prevSign = 0;
    this.prevUpdate = 1;
    this.adaGrad = (rProp) ? false : adaGrad;
    this.momentum = (rProp) ? false : momentum;
  }
  
  public void setBatchUpdate(boolean b) {
    this.batchUpdate = b;
    this.batchSum = 0;
  }

  public void reinitializeWeight(double low, double high) {
    weight = initializeWeight(low, high);
  }

  private double initializeWeight(double low, double high) {
    return Math.random() * (high - low) + low;
  }

  public void setLearningRate(double rate) {
    this.learningRate = rate;
  }

  public double getLearningRate() {
    return learningRate;
  }

  public void setIncomingElement(NetworkElement element) {
    this.incoming = element;
  }

  public NetworkElement getIncomingElement() {
    return incoming;
  }

  public void setOutgoingElement(NetworkElement element) {
    this.outgoing = element;
  }

  public NetworkElement getOutgoingElement() {
    return outgoing;
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
    setGradient(outgoing.getGradient() * derivative());
    updateWeight();
  }

  protected void updateWeight() {
    prevSign = (prevSign == 0) ? sign(gradient) : prevSign;
    if (batchUpdate)
      batchSum += gradient;
    else
      weight = weight - getUpdate();
  }

  /* Adagrad is for stochastic grad descent. We don't use it here */
  public void batchUpdate() {
    if (batchUpdate) {
      double update = (learningRate * batchSum);
      if(rProp) 
        update = rPropUpdate(batchSum);
      else if(momentum)
        update = addMomentum(update);
      weight = weight - update;
      prevUpdate = update;
      prevSign = sign(batchSum);
      batchSum = 0;
    }
  }
  
  /* Only ever called for SGD */
  private double getUpdate() {
    double update;
    if(adaGrad) {
      update = (newLearningRate() * gradient);
    } else if(rProp) {
      update = rPropUpdate(gradient);
    } else {
      update = learningRate * gradient;
    }
    if(momentum)
      update = addMomentum(update);
    prevUpdate = update;
    prevSign = sign(gradient);
    return update;
  }
  
  private double rPropUpdate(double grad) {
    return (sign(grad) != prevSign) ? (prevUpdate * nMinus) : (prevUpdate * nPlus);
  }
  
  private int sign(double val) {
    return (val >= 0) ? 1 : -1;
  }
  
  private double newLearningRate() {
    adaGradientSum += Math.pow(gradient, 2);
    double denom = Math.sqrt(Math.max(adaGradientSum, eps));
    double newLearningRate = learningRate / denom;
    return newLearningRate;
  }
  
  private double addMomentum(double update) {
    double momentum = alpha * prevUpdate;
    return update + momentum;
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

  public void setGradient(double g) {
    gradient = g;
  }

}

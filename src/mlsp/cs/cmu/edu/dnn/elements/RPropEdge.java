package mlsp.cs.cmu.edu.dnn.elements;

public class RPropEdge extends Edge {

  private static final long serialVersionUID = 7621029224723256343L;

  private boolean initialized = false;

  private double nPlus = 1.2;

  private double nMinus = 0.5;

  private int prevSign = 1;

  private double prevUpdate;

  public RPropEdge(double initalStep) {
    super();
    this.prevUpdate = initalStep;
  }

  public RPropEdge(double low, double high, double initalStep) {
    super(low, high, 0);
    this.prevUpdate = initalStep;
  }

  @Override
  public void updateWeight() {
    if (!isBatchUpdate()) {
      if (!initialized) {
        initialized = true;
        prevUpdate *= sign(getGradient());
        prevSign *= sign(getGradient());
      }
      int currSign = sign(getGradient());
      double update = prevUpdate * ((currSign == prevSign) ? nPlus : nMinus);
      prevUpdate = update;
      prevSign = currSign;
      double w = getWeight() - update;
      setWeight(w);
    }
  }

  @Override
  public void batchUpdate() {
    if (!initialized) {
      initialized = true;
      prevUpdate *= sign(getBatchGradient());
      prevSign *= sign(getBatchGradient());
    }
    int currSign = sign(getBatchGradient());
    double update = prevUpdate * ((currSign == prevSign) ? nPlus : nMinus);
    prevUpdate = update;
    prevSign = currSign;
    double w = getWeight() - update;
    setWeight(w);
  }
  
 

  private int sign(double d) {
    return (d >= 0) ? 1 : -1;
  }

}

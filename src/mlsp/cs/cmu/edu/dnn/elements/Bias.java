package mlsp.cs.cmu.edu.dnn.elements;

public class Bias extends Neuron {

  private static final long serialVersionUID = 8101699393546919528L;

  @Override
  public double getOutput() {
    return 1.0;
  }

  @Override
  public void forward() {
    /* nothing to do here... */
  }

  @Override
  public void backward() {
    /* nothing to do here... */
  }

  @Override
  public double derivative() {
    return 0;
  }

  @Override
  public double getErrorTerm() {
    return 0;
  }
  
}

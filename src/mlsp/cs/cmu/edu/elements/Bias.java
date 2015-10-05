package mlsp.cs.cmu.edu.elements;

public class Bias extends Neuron {

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

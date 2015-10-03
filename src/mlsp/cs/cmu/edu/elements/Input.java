package mlsp.cs.cmu.edu.elements;

public class Input implements NetworkElement {

  private volatile double inputValue = 0;
  private double output;
  
  public void setInputValue(double val) {
    inputValue = val;
  }

  @Override
  public void forward() {
    output = inputValue;
  }

  @Override
  public void backward() {
    /* Nothing to do here... */
  }

  @Override
  public double derivative() {
    return 0;
  }

  @Override
  public double getOutput() {
    return output;
  }

  @Override
  public double getErrorTerm() {
    return 0;
  }

}

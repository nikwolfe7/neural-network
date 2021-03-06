package mlsp.cs.cmu.edu.dnn.elements;

public class Input extends Neuron {

  private static final long serialVersionUID = -820683121797459764L;
  
  private double inputValue = 0;
  private double output = 0;
  
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
  public double getGradient() {
    return 0;
  }

}
